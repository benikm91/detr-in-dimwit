"""The `benikm91/l-shape <https://huggingface.co/datasets/benikm91/l-shape>`_
drawings, as the arrays DimWit lifts them from. Everything else happens in Scala.

The dataset ships its splits as plain repository files rather than as a
``datasets`` config: ``{split}_images.npy`` holds ``(N, 256, 256)`` uint8
drawings (row index = y, white background, dark ink) and
``{split}_labels.jsonl`` one drawing program per line.

A drawing program is parsed into the record it draws -- the drawn nodes first,
in the order they are drawn, then the relationships between them in a canonical
order, so that a record read back out of an adjacency matrix is the record it
came from. Everything else (``HelpLine``, ``BothSidedArrow``, ``FinishDrawing``)
is rendering, not record.
"""

import json

import numpy as np
from huggingface_hub import hf_hub_download

REPO_ID = "benikm91/l-shape"


def drawings(split):
    """The images of a split, memory mapped so that only what is read is read."""
    return np.load(_file(split, "images.npy"), mmap_mode="r")


def records(split, nodes, no_node, line, annotation, connected, annotates):
    """``(node_class, xs, ys, links)`` of every drawing, padded to ``nodes`` nodes."""
    with open(_file(split, "labels.jsonl"), encoding="utf-8") as programs:
        parsed = [_record_of(_actions(program), line, annotation, connected, annotates) for program in programs]

    node_class = np.full((len(parsed), nodes), no_node, dtype=np.int32)
    xs = np.zeros((len(parsed), nodes, 2), dtype=np.float32)
    ys = np.zeros((len(parsed), nodes, 2), dtype=np.float32)
    links = np.zeros((len(parsed), nodes, 2), dtype=np.int32)

    for drawing, (drawn, related) in enumerate(parsed):
        if len(drawn) + len(related) > nodes:
            raise ValueError("a record of %d nodes does not fit in %d" % (len(drawn) + len(related), nodes))
        for at, (node, node_xs, node_ys) in enumerate(drawn):
            node_class[drawing, at], xs[drawing, at, : len(node_xs)], ys[drawing, at, : len(node_ys)] = node, node_xs, node_ys
        for at, (relationship, subject, obj) in enumerate(related, start=len(drawn)):
            node_class[drawing, at], links[drawing, at] = relationship, (subject, obj)

    return node_class, xs, ys, links


def _record_of(actions, line, annotation, connected, annotates):
    """The ``(class, xs, ys)`` nodes and ``(class, subject, object)`` relationships of one program."""
    nodes, related, lines = [], [], []
    for action in actions:
        if action["type"] == "PartLineWithId":
            # A line is undirected, so its end points go in ascending order along the axis it
            # runs, which is the order its box hands them back in.
            (ax, ay), (bx, by) = action["coordinates_params"]
            along = 0 if abs(bx - ax) >= abs(by - ay) else 1
            (x1, y1), (x2, y2) = sorted(action["coordinates_params"], key=lambda point: point[along])
            lines.append(len(nodes))
            nodes.append((line, (x1, x2), (y1, y2)))
        elif action["type"] == "AnnotationTextRefId":
            ((x, y),) = action["coordinates_params"]
            nodes.append((annotation, (x,), (y,)))
            related.append((annotates, len(nodes) - 1, lines[int(action["discrete_params"][-1])]))
        elif action["type"] == "ConnectTwoElementsWithId":
            # Undirected too, so the corner is held once with the two it links in ascending order.
            related.append((connected, *sorted(lines[int(end)] for end in action["discrete_params"][-2:])))
    return nodes, sorted(related)


def _actions(program):
    actions = json.loads(program)["actions"]
    return json.loads(actions) if isinstance(actions, str) else actions


def _file(split, name):
    return hf_hub_download(repo_id=REPO_ID, filename="%s_%s" % (split, name), repo_type="dataset")
