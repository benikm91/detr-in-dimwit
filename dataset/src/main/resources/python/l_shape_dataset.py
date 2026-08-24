"""Python side of the DimWit ``LShapeDataset`` wrapper.

Loads the `benikm91/l-shape <https://huggingface.co/datasets/benikm91/l-shape>`_
dataset through ``huggingface_hub`` and hands out the *record* of every drawing:
the graph it was rendered from, which is what the drawing program spells out.

The dataset is stored in the ``npy-memmap-v1`` format, i.e. as plain files in
the repository rather than as a ``datasets`` config:

* ``{split}_images.npy``  -- ``uint8`` array of shape ``(N, 256, 256)``,
  row index = y, column index = x, white (255) background, dark (0) ink.
* ``{split}_labels.jsonl`` -- one object per line, ``{"index": i, "actions": "<json>"}``
  where ``actions`` is a *string* holding the drawing program for that image.
* ``{split}_seeds.npy``   -- generator seeds (unused here).

A record is a list of nodes. The drawn ones come first, in the order the drawing
program draws them, followed by the relationships between them in a canonical
order, so that a record read back out of an adjacency matrix is the record it
came from:

* ``PartLineWithId``           -- a line node, by its two end points, in ascending order along
  the axis the line runs, since a line is undirected;
* ``AnnotationTextRefId``      -- an annotation node, by its centre, plus an
  ``annotates`` relationship to the line it measures;
* ``ConnectTwoElementsWithId`` -- a ``connected`` relationship between two
  lines, held once with the two it links in ascending order.

Everything else (``HelpLine``, ``BothSidedArrow``, ``FinishDrawing``) is
rendering, not record.

Since the training split is ~8.6 GB the images are memory mapped and only the
requested samples are ever read; parsed records are small and are cached on disk
as an ``.npz`` so that the ~131k line JSONL is parsed only once.
"""

from __future__ import annotations

import hashlib
import json
import os

import numpy as np

DEFAULT_REPO_ID = "benikm91/l-shape"

#: Points a node can be placed by: the two end points of a line.
POINTS = 2

#: Nodes a relationship can link: its subject and its object.
LINKS = 2


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


def _record_of(actions, classes):
    """The record of one drawing program.

    Returns ``(nodes, relationships, unresolved)`` where a node is
    ``(class_id, xs, ys)`` and a relationship is ``(class_id, subject, object)``,
    both naming nodes by the position they end up in.
    """
    nodes = []
    relationships = []
    slot_of_element = {}
    next_element_id = 0
    unresolved = 0

    def slot_of(param):
        try:
            return slot_of_element.get(int(param))
        except (TypeError, ValueError):
            return None

    for action in actions:
        action_type = action["type"]
        if action_type == "PartLineWithId":
            # A line is undirected, so its end points are held in ascending order along the
            # axis it runs, which is the order its box hands them back in.
            (ax, ay), (bx, by) = action["coordinates_params"]
            along = 0 if abs(bx - ax) >= abs(by - ay) else 1
            (x1, y1), (x2, y2) = sorted(action["coordinates_params"], key=lambda point: point[along])
            slot_of_element[next_element_id] = len(nodes)
            next_element_id += 1
            nodes.append((classes["line"], (x1, x2), (y1, y2)))
        elif action_type == "AnnotationTextRefId":
            (x, y), = action["coordinates_params"]
            annotation = len(nodes)
            nodes.append((classes["annotation"], (x,), (y,)))
            line = slot_of(action.get("discrete_params", [None])[-1])
            if line is None:
                unresolved += 1
            else:
                relationships.append((classes["annotates"], annotation, line))
        elif action_type == "ConnectTwoElementsWithId":
            ends = [slot_of(param) for param in action.get("discrete_params", [])[-2:]]
            if len(ends) < 2 or None in ends:
                unresolved += 1
            else:
                relationships.append((classes["connected"], min(ends), max(ends)))

    return nodes, sorted(relationships), unresolved


def _parse_records(labels_path, classes, max_num_nodes):
    """Parse the JSONL label file into padded ``(node_class, xs, ys, links)``."""
    records = []
    unresolved = 0
    with open(labels_path, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            actions = json.loads(line)["actions"]
            if isinstance(actions, str):  # the actions column is a JSON string
                actions = json.loads(actions)
            nodes, relationships, sample_unresolved = _record_of(actions, classes)
            records.append((nodes, relationships))
            unresolved += sample_unresolved

    observed_max = max((len(nodes) + len(edges) for nodes, edges in records), default=0)
    width = max_num_nodes if max_num_nodes > 0 else max(observed_max, 1)

    node_class = np.full((len(records), width), classes["no_node"], dtype=np.int32)
    xs = np.zeros((len(records), width, POINTS), dtype=np.float32)
    ys = np.zeros((len(records), width, POINTS), dtype=np.float32)
    links = np.zeros((len(records), width, LINKS), dtype=np.int32)
    truncated = 0

    for sample, (nodes, relationships) in enumerate(records):
        for at, (node_class_id, node_xs, node_ys) in enumerate(nodes[:width]):
            node_class[sample, at] = node_class_id
            xs[sample, at, : len(node_xs)] = node_xs
            ys[sample, at, : len(node_ys)] = node_ys
        for offset, relationship in enumerate(relationships):
            at = len(nodes) + offset
            if at >= width or max(relationship[1:]) >= len(nodes[:width]):
                truncated += 1
                continue
            node_class[sample, at] = relationship[0]
            links[sample, at] = relationship[1:]
        truncated += max(0, len(nodes) - width)

    return node_class, xs, ys, links, observed_max, truncated, unresolved


def load_records(labels_path, classes, max_num_nodes=0, use_cache=True):
    """Parsed records for a split, memoized on disk."""
    identity = json.dumps(
        {
            "path": os.path.realpath(labels_path),
            "size": os.path.getsize(labels_path),
            "classes": sorted(classes.items()),
            "nodes": int(max_num_nodes),
            "version": 6,
        },
        sort_keys=True,
    )
    cache = _cache_file("records-" + hashlib.sha1(identity.encode("utf-8")).hexdigest()[:16])

    if use_cache and os.path.exists(cache):
        try:
            with np.load(cache) as cached:
                return (
                    cached["node_class"],
                    cached["xs"],
                    cached["ys"],
                    cached["links"],
                    int(cached["observed_max"]),
                    int(cached["truncated"]),
                    int(cached["unresolved"]),
                )
        except (OSError, ValueError, KeyError):
            pass  # corrupt or outdated cache: fall through and re-parse

    parsed = _parse_records(labels_path, classes, max_num_nodes)
    node_class, xs, ys, links, observed_max, truncated, unresolved = parsed

    if use_cache:
        staging = "%s.%d.tmp.npz" % (cache, os.getpid())  # savez keeps an .npz suffix as is
        np.savez(
            staging,
            node_class=node_class,
            xs=xs,
            ys=ys,
            links=links,
            observed_max=np.int64(observed_max),
            truncated=np.int64(truncated),
            unresolved=np.int64(unresolved),
        )
        os.replace(staging, cache)

    return node_class, xs, ys, links, observed_max, truncated, unresolved


class Records:
    """Images plus the records they were rendered from, as JAX arrays.

    ``images`` is ``(width, height, channel)`` for a single sample and
    ``(sample, width, height, channel)`` for a batch; the record arrays gain the
    same leading axis.
    """

    __slots__ = ("images", "node_class", "xs", "ys", "links")

    def __init__(self, images, node_class, xs, ys, links):
        self.images = images
        self.node_class = node_class
        self.xs = xs
        self.ys = ys
        self.links = links


class Split:
    """One split of the l-shape dataset, with memory mapped images."""

    def __init__(
        self,
        repo_id,
        split,
        revision="",
        no_node_class=0,
        line_class=1,
        annotation_class=2,
        connected_class=3,
        annotates_class=4,
        max_num_nodes=0,
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

        classes = {
            "no_node": int(no_node_class),
            "line": int(line_class),
            "annotation": int(annotation_class),
            "connected": int(connected_class),
            "annotates": int(annotates_class),
        }
        (
            self.node_class,
            self.xs,
            self.ys,
            self.links,
            self.observed_max_nodes,
            self.truncated_nodes,
            self.unresolved_links,
        ) = load_records(labels_path, classes, int(max_num_nodes), use_cache)

        if len(self.node_class) != self.num_samples:
            raise ValueError(
                "split '%s' has %d images but %d label rows"
                % (split, self.num_samples, len(self.node_class))
            )

        self.normalize = bool(normalize)
        self.max_num_nodes = int(self.node_class.shape[1])

    def _images(self, selection):
        """Read the selected images as ``(..., width, height, channel)`` float arrays.

        The stored layout is row major (row = y), so it is transposed to put the
        x axis first, matching the ``x``/``y`` order of a record's coordinates.
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

    def _records(self, selection):
        import jax.numpy as jnp

        return Records(
            jnp.asarray(self._images(selection)),
            jnp.asarray(self.node_class[selection]),
            jnp.asarray(self.xs[selection]),
            jnp.asarray(self.ys[selection]),
            jnp.asarray(self.links[selection]),
        )

    def sample(self, index):
        """A single sample, without a leading batch axis."""
        index = int(index)
        self._check(index)
        return self._records(index)

    def batch(self, indices):
        """Gather ``indices`` into one batch."""
        selection = np.asarray(indices, dtype=np.int64)
        if selection.size:
            self._check(int(selection.min()))
            self._check(int(selection.max()))
        return self._records(selection)


def open_split(
    repo_id=DEFAULT_REPO_ID,
    split="val",
    revision="",
    no_node_class=0,
    line_class=1,
    annotation_class=2,
    connected_class=3,
    annotates_class=4,
    max_num_nodes=0,
    normalize=True,
    use_cache=True,
):
    """Entry point used by the Scala wrapper."""
    return Split(
        repo_id=repo_id,
        split=split,
        revision=revision,
        no_node_class=no_node_class,
        line_class=line_class,
        annotation_class=annotation_class,
        connected_class=connected_class,
        annotates_class=annotates_class,
        max_num_nodes=max_num_nodes,
        normalize=normalize,
        use_cache=use_cache,
    )
