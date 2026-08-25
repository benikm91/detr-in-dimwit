"""What a record stands for, as a picture.

A record has no drawing of its own -- it is what a drawing encodes -- so a
transcription can only be looked at by drawing it: every line as the segment
between its end points, every annotation as a marker on its point, and every
relationship as a dashed connector between the two nodes it relates. The result
is drawn over the drawing the record was read from, so that the two can be
compared pixel by pixel.
"""

import numpy as np

#: Colours the parts of a record are drawn in, as ``(red, green, blue)``.
LINE = (20, 60, 190)
ANNOTATION = (230, 140, 20)
CONNECTED = (20, 160, 110)
ANNOTATES = (170, 70, 200)


def render(drawing, node_class, xs, ys, edge_class, links, line, annotation, connected, annotates):
    """``(width, height, 3)`` uint8 pixels of the record drawn over ``drawing``, which is a
    ``(width, height)`` grey level image of what it was read from."""
    drawing = np.asarray(drawing, dtype=np.uint8)
    image = np.repeat(drawing[:, :, None], 3, axis=2)
    node_class, edge_class = np.asarray(node_class, dtype=int), np.asarray(edge_class, dtype=int)
    xs, ys = np.asarray(xs, dtype=float).reshape(len(node_class), -1), np.asarray(ys, dtype=float).reshape(len(node_class), -1)
    links = np.asarray(links, dtype=int).reshape(len(edge_class), -1)

    def anchor(node):
        """Where a relationship reaches a node: the middle of a line, the point of an annotation."""
        if node_class[node] == line:
            return (xs[node][:2].mean(), ys[node][:2].mean())
        return (xs[node][0], ys[node][0])

    # The relationships go on first, so that the nodes they relate stay crisp on top of them.
    for at, held in enumerate(edge_class):
        if held in (connected, annotates):
            colour = CONNECTED if held == connected else ANNOTATES
            _segment(image, anchor(links[at][0]), anchor(links[at][1]), colour, dashed=True)

    for at, held in enumerate(node_class):
        if held == line:
            _segment(image, (xs[at][0], ys[at][0]), (xs[at][1], ys[at][1]), LINE)
        elif held == annotation:
            _dot(image, xs[at][0], ys[at][0], ANNOTATION, radius=2)

    return image


def _segment(image, start, end, colour, dashed=False):
    """A straight run of pixels from one normalized point to another.

    Two steps per pixel it spans, so that a slope never leaves a gap between them.
    """
    canvas = image.shape[0]
    (x0, y0), (x1, y1) = start, end
    steps = 2 * int(np.ceil(max(abs(x1 - x0), abs(y1 - y0)) * canvas)) + 1
    for step, (x, y) in enumerate(zip(np.linspace(x0, x1, steps), np.linspace(y0, y1, steps))):
        if not dashed or (step // 8) % 2 == 0:
            _dot(image, x, y, colour)


def _dot(image, x, y, colour, radius=0):
    """One normalized point, as the pixels within `radius` of where it falls."""
    canvas = image.shape[0]
    at_x, at_y = int(round(x * canvas)), int(round(y * canvas))
    within = lambda at: slice(max(at - radius, 0), min(at + radius + 1, canvas))
    image[within(at_x), within(at_y)] = colour
