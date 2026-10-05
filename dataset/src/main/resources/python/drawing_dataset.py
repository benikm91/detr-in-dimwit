"""The drawing datasets, as the arrays DimWit lifts them from. Everything else
happens in Scala.

Every corpus ships its splits as plain repository files rather than as a
``datasets`` config: ``{split}_images.npy`` holds ``(N, canvas, canvas)`` uint8
drawings (row index = y, white background, dark ink) and
``{split}_labels.jsonl`` one label per line, both in the corpus's folder of the
repository.

A label is one of two things. A drawing program, which the generated corpora are
written in, is parsed into the record it draws -- the drawn nodes in the order
they are drawn, and the relationships between them in a canonical order, so
that a record read back out of an adjacency matrix is the record it came from.
Everything else (``HelpLine``, ``BothSidedArrow``, ``FinishDrawing``) is
rendering, not record. A sketch, which the SketchGraphs corpus of Vitruvion is
written in, holds its primitives in the pixels of its drawing, and the constraints
between them.
"""

import json

import numpy as np
from huggingface_hub import hf_hub_download

def drawings(repo_id, folder, split):
    """The images of a split, memory mapped so that only what is read is read."""
    return np.load(_file(repo_id, folder, split, "images.npy"), mmap_mode="r")


def records(repo_id, folder, split, canvas, nodes, edges, no_node, line, annotation, circle, arc, point, no_edge, connected, annotates, edge_classes):
    """``(node_class, construction, start_x, start_y, end_x, end_y, mid_x, mid_y, edge_class,
    subject, obj)`` of every drawing, padded to ``nodes`` drawn nodes and ``edges`` relationships.

    A node is placed by where it starts, if it runs somewhere by where it ends, and if it bends by
    its middle, as a fraction of the ``canvas`` side; what a node does not place is left at zero.
    ``edge_classes`` names every relationship class by its id. A corpus with no room for
    relationships reads a sketch's primitives alone.
    """
    edge_class_ids = {name: at for at, name in enumerate(edge_classes)}

    def record_of(label):
        held = json.loads(label)
        if "primitives" in held:
            constraints = held["constraints"] if edges else []
            return _record_of_sketch(held["primitives"], constraints, canvas, line, circle, arc, point, edge_class_ids)
        return _record_of(_actions(held), line, annotation, circle, arc, connected, annotates)

    with open(_file(repo_id, folder, split, "labels.jsonl"), encoding="utf-8") as labels:
        parsed = [record_of(label) for label in labels]

    node_class = np.full((len(parsed), nodes), no_node, dtype=np.int32)
    construction = np.zeros((len(parsed), nodes), dtype=np.int32)
    start_x, start_y = (np.zeros((len(parsed), nodes), dtype=np.float32) for _ in range(2))
    end_x, end_y = (np.zeros((len(parsed), nodes), dtype=np.float32) for _ in range(2))
    mid_x, mid_y = (np.zeros((len(parsed), nodes), dtype=np.float32) for _ in range(2))
    edge_class = np.full((len(parsed), edges), no_edge, dtype=np.int32)
    subject, obj = (np.zeros((len(parsed), edges), dtype=np.int32) for _ in range(2))

    for drawing, (drawn, related) in enumerate(parsed):
        if len(drawn) > nodes:
            raise ValueError("a record of %d nodes does not fit in %d" % (len(drawn), nodes))
        if len(related) > edges:
            raise ValueError("a record of %d relationships does not fit in %d" % (len(related), edges))
        for at, (node, is_construction, node_xs, node_ys) in enumerate(drawn):
            node_class[drawing, at], construction[drawing, at] = node, is_construction
            start_x[drawing, at], start_y[drawing, at] = node_xs[0], node_ys[0]
            if len(node_xs) > 1:
                end_x[drawing, at], end_y[drawing, at] = node_xs[1], node_ys[1]
            if len(node_xs) > 2:
                mid_x[drawing, at], mid_y[drawing, at] = node_xs[2], node_ys[2]
        for at, (relationship, relates, to) in enumerate(related):
            edge_class[drawing, at], subject[drawing, at], obj[drawing, at] = relationship, relates, to

    return node_class, construction, start_x, start_y, end_x, end_y, mid_x, mid_y, edge_class, subject, obj


def _record_of(actions, line, annotation, circle, arc, connected, annotates):
    """The ``(class, construction, xs, ys)`` nodes and ``(class, subject, object)`` relationships of
    one program, which draws no construction geometry."""
    nodes, related, lines = [], [], []
    for action in actions:
        if action["type"] == "PartLineWithId":
            # A line is undirected, so its end points go in ascending order along the axis it
            # runs, which is the order its box hands them back in.
            (ax, ay), (bx, by) = action["coordinates_params"]
            along = 0 if abs(bx - ax) >= abs(by - ay) else 1
            (x1, y1), (x2, y2) = sorted(action["coordinates_params"], key=lambda point: point[along])
            lines.append(len(nodes))
            nodes.append((line, False, (x1, x2), (y1, y2)))
        elif action["type"] == "Circle":
            # The two ends of the horizontal diameter, leftmost first, which is how a record holds
            # a circle and how the corpus already writes it.
            (x1, y1), (x2, y2) = action["coordinates_params"]
            nodes.append((circle, False, (x1, x2), (y1, y2)))
        elif action["type"] == "Arc":
            # Start and end first, where a record holds every node's, and the middle after them.
            # The corpus already writes an arc clockwise, which is what makes its start its start.
            (sx, sy), (mx, my), (ex, ey) = action["coordinates_params"]
            nodes.append((arc, False, (sx, ex, mx), (sy, ey, my)))
        elif action["type"] == "AnnotationTextRefId":
            ((x, y),) = action["coordinates_params"]
            nodes.append((annotation, False, (x,), (y,)))
            related.append((annotates, len(nodes) - 1, lines[int(action["discrete_params"][-1])]))
        elif action["type"] == "ConnectTwoElementsWithId":
            # Undirected too, so the corner is held once with the two it links in ascending order.
            related.append((connected, *sorted(lines[int(end)] for end in action["discrete_params"][-2:])))
    return nodes, sorted(related)


def _record_of_sketch(primitives, constraints, canvas, line, circle, arc, point, edge_class_ids):
    """The ``(class, construction, xs, ys)`` nodes of one sketch, held as a record holds them, and
    the ``(class, subject, object)`` relationships its constraints are.

    A constraint names the part of each node it takes hold of, and that part goes into its class:
    ``coincident end-start`` holds the end of its subject on the start of its object. One holding a
    single node relates that node to itself. One holding more, a midpoint between two points, is no
    relationship between two and is left out.
    """
    nodes, swapped = [], []
    for primitive in primitives:
        held = [(x / canvas, y / canvas) for x, y in primitive["points"]]
        if primitive["type"] == "line":
            # In ascending order along the axis it runs, as a drawn line's ends are.
            (ax, ay), (bx, by) = held
            along = 0 if abs(bx - ax) >= abs(by - ay) else 1
            (x1, y1), (x2, y2) = sorted(held, key=lambda at: at[along])
            nodes.append((line, primitive["construction"], (x1, x2), (y1, y2)))
            swapped.append(held[0][along] > held[1][along])
        elif primitive["type"] == "circle":
            ((x, y),), radius = held, primitive["radius"] / canvas
            nodes.append((circle, primitive["construction"], (x - radius, x + radius), (y, y)))
            swapped.append(False)
        elif primitive["type"] == "arc":
            # A sketch reads an arc counter-clockwise on the drawing, a record clockwise.
            (sx, sy), (mx, my), (ex, ey) = held
            nodes.append((arc, primitive["construction"], (ex, sx, mx), (ey, sy, my)))
            swapped.append(True)
        else:
            ((x, y),) = held
            nodes.append((point, primitive["construction"], (x,), (y,)))
            swapped.append(False)

    def part(node, named):
        """The part of a node a constraint names, as the record holds the node."""
        if named in ("start", "end") and swapped[node]:
            return "end" if named == "start" else "start"
        return {None: "whole", "center": "centre"}.get(named, named)

    related = []
    for constraint in constraints:
        # A few constraints name the same part twice, which holds it no more than once.
        refs = list(dict.fromkeys(tuple(ref) for ref in constraint["refs"]))
        if len(refs) > 2:
            continue
        (subject, held_by_subject), (obj, held_by_obj) = refs if len(refs) == 2 else refs * 2
        name = "%s %s-%s" % (constraint["type"], part(subject, held_by_subject), part(obj, held_by_obj))
        related.append((edge_class_ids[name], subject, obj))
    return nodes, sorted(related)


def _actions(held):
    actions = held["actions"]
    return json.loads(actions) if isinstance(actions, str) else actions


def _file(repo_id, folder, split, name):
    filename = "%s_%s" % (split, name)
    return hf_hub_download(repo_id=repo_id, filename="%s/%s" % (folder, filename) if folder else filename, repo_type="dataset")
