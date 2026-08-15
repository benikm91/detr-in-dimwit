"""Python side of the DimWit ``LShapeDataset`` wrapper.

Loads the `benikm91/l-shape <https://huggingface.co/datasets/benikm91/l-shape>`_
dataset through ``huggingface_hub`` and exposes it in a shape that is directly
liftable into DimWit tensors.

The dataset is stored in the ``npy-memmap-v1`` format, i.e. as plain files in
the repository rather than as a ``datasets`` config:

* ``{split}_images.npy``  -- ``uint8`` array of shape ``(N, 256, 256)``,
  row index = y, column index = x, white (255) background, dark (0) ink.
* ``{split}_labels.jsonl`` -- one object per line, ``{"index": i, "actions": "<json>"}``
  where ``actions`` is a *string* holding the drawing program for that image.
* ``{split}_seeds.npy``   -- generator seeds (unused here).

A drawing program is a graph: nodes carry coordinates, and relationship actions
(``ConnectTwoElementsWithId``) tie them together. Object detection only cares
about the nodes, so every action of a requested type becomes one axis aligned
box in normalized ``(cx, cy, w, h)`` coordinates -- the convention DETR predicts
in -- and everything else is dropped. All drawn lines are straight and axis
aligned, so no rotated boxes are needed:

* an action with two points spans a box between them, grown to at least
  ``min_width``/``min_height`` so that a horizontal or vertical line keeps a
  non-degenerate box;
* an action with a single point (the text anchor) becomes a box of the fixed
  size ``fixed_width``/``fixed_height`` centred on it.

Since the training split is ~8.6 GB the images are memory mapped and only the
requested samples are ever read; parsed targets are small and are cached on disk
as an ``.npz`` so that the ~131k line JSONL is parsed only once.
"""

from __future__ import annotations

import hashlib
import json
import os

import numpy as np

DEFAULT_REPO_ID = "benikm91/l-shape"

#: Class id reserved for padded query slots.
NO_OBJECT = 0


def _download(repo_id, filename, revision):
    """Resolve ``filename`` of a dataset repo to a local path, downloading if needed."""
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:  # pragma: no cover - depends on the user's env
        raise ImportError(
            "The l-shape dataset wrapper needs the 'huggingface_hub' package. "
            "Add 'huggingface-hub' to pyproject.toml and re-run 'uv sync'."
        ) from exc
    return hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        repo_type="dataset",
        revision=revision or None,
    )


def metadata(repo_id=DEFAULT_REPO_ID, revision=""):
    """Return the dataset's ``metadata.json`` as a dict."""
    with open(_download(repo_id, "metadata.json", revision), encoding="utf-8") as handle:
        return json.load(handle)


def _cache_file(key):
    root = os.environ.get("LSHAPE_CACHE_DIR") or os.path.join(
        os.environ.get("XDG_CACHE_HOME") or os.path.expanduser("~/.cache"),
        "dimwit-l-shape",
    )
    os.makedirs(root, exist_ok=True)
    return os.path.join(root, key + ".npz")


def _objects_of(actions, geometry):
    """Turn one drawing program into ``(cx, cy, w, h, class_id)`` rows."""
    rows = []
    for action in actions:
        box = geometry.get(action["type"])
        points = action.get("coordinates_params")
        if box is None or not points:
            continue
        class_id, min_width, min_height, fixed_width, fixed_height = box
        xs = [point[0] for point in points]
        ys = [point[1] for point in points]
        x_min, x_max = min(xs), max(xs)
        y_min, y_max = min(ys), max(ys)
        if len(points) == 1:
            width, height = fixed_width, fixed_height
        else:
            width = max(x_max - x_min, min_width)
            height = max(y_max - y_min, min_height)
        rows.append((0.5 * (x_min + x_max), 0.5 * (y_min + y_max), width, height, class_id))
    return rows


def _parse_labels(labels_path, geometry, max_num_objects):
    """Parse the JSONL label file into padded ``(boxes, labels)`` arrays."""
    rows = []
    counts = []
    with open(labels_path, encoding="utf-8") as handle:
        for sample_index, line in enumerate(handle):
            line = line.strip()
            if not line:
                continue
            actions = json.loads(line)["actions"]
            if isinstance(actions, str):  # the actions column is a JSON string
                actions = json.loads(actions)
            sample_rows = _objects_of(actions, geometry)
            rows.extend((sample_index,) + row for row in sample_rows)
            counts.append(len(sample_rows))

    num_samples = len(counts)
    counts = np.asarray(counts, dtype=np.int64)
    observed_max = int(counts.max()) if num_samples else 0
    slots = max_num_objects if max_num_objects > 0 else max(observed_max, 1)

    boxes = np.zeros((num_samples, slots, 4), dtype=np.float32)
    labels = np.full((num_samples, slots), NO_OBJECT, dtype=np.int32)
    truncated = 0

    if rows:
        table = np.asarray(rows, dtype=np.float64)
        sample_of_row = table[:, 0].astype(np.int64)
        # Rows are emitted in sample order, so the slot of a row is its offset
        # from the first row belonging to the same sample.
        starts = np.concatenate(([0], np.cumsum(counts)[:-1]))
        row_slots = np.arange(len(table), dtype=np.int64) - starts[sample_of_row]
        keep = row_slots < slots
        boxes[sample_of_row[keep], row_slots[keep]] = table[keep, 1:5]
        labels[sample_of_row[keep], row_slots[keep]] = table[keep, 5].astype(np.int32)
        truncated = int((~keep).sum())

    return boxes, labels, observed_max, truncated


def load_targets(labels_path, geometry, max_num_objects=0, use_cache=True):
    """Parsed detection targets for a split, memoized on disk.

    Returns ``(boxes, labels, observed_max_objects, truncated)`` where ``boxes``
    has shape ``(N, slots, 4)`` in ``(cx, cy, w, h)`` order and ``labels`` shape
    ``(N, slots)``.
    """
    identity = json.dumps(
        {
            "path": os.path.realpath(labels_path),
            "size": os.path.getsize(labels_path),
            "geometry": sorted(geometry.items()),
            "slots": int(max_num_objects),
            "version": 2,
        },
        sort_keys=True,
    )
    cache = _cache_file("targets-" + hashlib.sha1(identity.encode("utf-8")).hexdigest()[:16])

    if use_cache and os.path.exists(cache):
        try:
            with np.load(cache) as cached:
                return (
                    cached["boxes"],
                    cached["labels"],
                    int(cached["observed_max"]),
                    int(cached["truncated"]),
                )
        except (OSError, ValueError, KeyError):
            pass  # corrupt or outdated cache: fall through and re-parse

    boxes, labels, observed_max, truncated = _parse_labels(labels_path, geometry, max_num_objects)

    if use_cache:
        staging = "%s.%d.tmp.npz" % (cache, os.getpid())  # savez keeps an .npz suffix as is
        np.savez(
            staging,
            boxes=boxes,
            labels=labels,
            observed_max=np.int64(observed_max),
            truncated=np.int64(truncated),
        )
        os.replace(staging, cache)

    return boxes, labels, observed_max, truncated


class Objects:
    """Images plus their detection targets, as JAX arrays.

    ``images`` is ``(width, height, channel)`` for a single sample and
    ``(sample, width, height, channel)`` for a batch; the target arrays gain the
    same leading axis.
    """

    __slots__ = ("images", "center_x", "center_y", "width", "height", "label")

    def __init__(self, images, center_x, center_y, width, height, label):
        self.images = images
        self.center_x = center_x
        self.center_y = center_y
        self.width = width
        self.height = height
        self.label = label


class Split:
    """One split of the l-shape dataset, with memory mapped images."""

    def __init__(
        self,
        repo_id,
        split,
        revision="",
        class_names=(),
        class_ids=(),
        max_num_objects=0,
        min_size_pixels=4.0,
        text_size_pixels=12.0,
        normalize=True,
        use_cache=True,
    ):
        images_path = _download(repo_id, "%s_images.npy" % split, revision)
        labels_path = _download(repo_id, "%s_labels.jsonl" % split, revision)

        self.repo_id = repo_id
        self.split = split
        self.images = np.load(images_path, mmap_mode="r")
        self.num_samples = int(self.images.shape[0])
        self.image_height = int(self.images.shape[1])
        self.image_width = int(self.images.shape[2])

        # Box sizes are given in pixels but stored normalized, like the coordinates.
        min_width = min_size_pixels / self.image_width
        min_height = min_size_pixels / self.image_height
        fixed_width = text_size_pixels / self.image_width
        fixed_height = text_size_pixels / self.image_height
        geometry = {
            name: (int(class_id), min_width, min_height, fixed_width, fixed_height)
            for name, class_id in zip(class_names, class_ids)
        }

        self.boxes, self.labels, self.observed_max_objects, self.truncated_objects = load_targets(
            labels_path, geometry, max_num_objects, use_cache
        )

        if len(self.labels) != self.num_samples:
            raise ValueError(
                "split '%s' has %d images but %d label rows"
                % (split, self.num_samples, len(self.labels))
            )

        self.normalize = bool(normalize)
        self.max_num_objects = int(self.labels.shape[1])

    def _images(self, selection):
        """Read the selected images as ``(..., width, height, channel)`` float arrays.

        The stored layout is row major (row = y), so it is transposed to put the
        x axis first -- matching DETR's ``Tensor3[Width, Height, Channel]`` and
        the ``cx``/``cy`` targets.
        """
        images = np.asarray(self.images[selection])  # only these rows are read
        images = np.swapaxes(images, -2, -1)[..., None].astype(np.float32)
        if self.normalize:
            images /= 255.0
        return images

    def _check(self, index):
        if index < 0 or index >= self.num_samples:
            raise IndexError(
                "index %d out of range for split '%s' with %d samples"
                % (index, self.split, self.num_samples)
            )

    def sample(self, index):
        """A single sample, without a leading batch axis."""
        import jax.numpy as jnp

        index = int(index)
        self._check(index)
        boxes = self.boxes[index]
        return Objects(
            jnp.asarray(self._images(index)),
            jnp.asarray(boxes[:, 0]),
            jnp.asarray(boxes[:, 1]),
            jnp.asarray(boxes[:, 2]),
            jnp.asarray(boxes[:, 3]),
            jnp.asarray(self.labels[index]),
        )

    def batch(self, indices):
        """Gather ``indices`` into one batch."""
        import jax.numpy as jnp

        selection = np.asarray(indices, dtype=np.int64)
        if selection.size:
            self._check(int(selection.min()))
            self._check(int(selection.max()))
        boxes = self.boxes[selection]
        return Objects(
            jnp.asarray(self._images(selection)),
            jnp.asarray(boxes[:, :, 0]),
            jnp.asarray(boxes[:, :, 1]),
            jnp.asarray(boxes[:, :, 2]),
            jnp.asarray(boxes[:, :, 3]),
            jnp.asarray(self.labels[selection]),
        )


def open_split(
    repo_id=DEFAULT_REPO_ID,
    split="val",
    revision="",
    class_names=(),
    class_ids=(),
    max_num_objects=0,
    min_size_pixels=4.0,
    text_size_pixels=12.0,
    normalize=True,
    use_cache=True,
):
    """Entry point used by the Scala wrapper."""
    return Split(
        repo_id=repo_id,
        split=split,
        revision=revision,
        class_names=class_names,
        class_ids=class_ids,
        max_num_objects=max_num_objects,
        min_size_pixels=min_size_pixels,
        text_size_pixels=text_size_pixels,
        normalize=normalize,
        use_cache=use_cache,
    )
