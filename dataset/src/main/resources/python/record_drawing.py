"""What a record stands for, as a picture.

A record has no drawing of its own -- it is what a drawing encodes -- so a
transcription can only be looked at by drawing it: every line as the segment
between its end points, every circle around the diameter its points span, every
arc through its three points, every annotation and every point as a marker on
its point, construction geometry dashed, and every relationship as a dashed
connector between the two nodes it relates. The result
is drawn over the drawing the record was read from, so that the two can be
compared pixel by pixel.
"""

import numpy as np

#: Colours the parts of a record are drawn in, as ``(red, green, blue)``.
LINE = (20, 60, 190)
CIRCLE = (20, 130, 200)
ARC = (200, 40, 60)
ANNOTATION = (230, 140, 20)
POINT = (120, 40, 160)
CONNECTED = (20, 160, 110)
ANNOTATES = (170, 70, 200)


def render(drawing, node_class, construction, start_x, start_y, end_x, end_y, mid_x, mid_y, edge_class, subject, obj, line, annotation, circle, arc, point, connected, annotates):
    """``(width, height, 3)`` uint8 pixels of the record drawn over ``drawing``, which is a
    ``(width, height)`` grey level image of what it was read from."""
    drawing = np.asarray(drawing, dtype=np.uint8)
    image = np.repeat(drawing[:, :, None], 3, axis=2)
    node_class, edge_class = np.asarray(node_class, dtype=int), np.asarray(edge_class, dtype=int)
    start_x, start_y = np.asarray(start_x, dtype=float), np.asarray(start_y, dtype=float)
    end_x, end_y = np.asarray(end_x, dtype=float), np.asarray(end_y, dtype=float)
    mid_x, mid_y = np.asarray(mid_x, dtype=float), np.asarray(mid_y, dtype=float)
    subject, obj = np.asarray(subject, dtype=int), np.asarray(obj, dtype=int)

    def anchor(node):
        """Where a relationship reaches a node: the middle of a line, a circle or an arc, the
        point of an annotation."""
        if node_class[node] == arc:
            return (mid_x[node], mid_y[node])
        if node_class[node] in (line, circle):
            return ((start_x[node] + end_x[node]) / 2, (start_y[node] + end_y[node]) / 2)
        return (start_x[node], start_y[node])

    # The relationships go on first, so that the nodes they relate stay crisp on top of them.
    for at, held in enumerate(edge_class):
        if held in (connected, annotates):
            colour = CONNECTED if held == connected else ANNOTATES
            _segment(image, anchor(subject[at]), anchor(obj[at]), colour, dashed=True)

    for at, held in enumerate(node_class):
        dashed = bool(construction[at])
        if held == line:
            _segment(image, (start_x[at], start_y[at]), (end_x[at], end_y[at]), LINE, dashed)
        elif held == circle:
            _ring(image, (start_x[at], start_y[at]), (end_x[at], end_y[at]), CIRCLE, dashed)
        elif held == arc:
            _bend(image, (start_x[at], start_y[at]), (mid_x[at], mid_y[at]), (end_x[at], end_y[at]), ARC, dashed)
        elif held == annotation:
            _dot(image, start_x[at], start_y[at], ANNOTATION, radius=2)
        elif held == point:
            _dot(image, start_x[at], start_y[at], POINT, radius=1)

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


def _ring(image, left, right, colour, dashed=False):
    """A circle, from the two ends of the horizontal diameter that place it.

    Two steps per pixel of its circumference, for the reason a segment takes two per pixel of its
    span: at one step a curve leaves gaps between them.
    """
    canvas = image.shape[0]
    (left_x, y), (right_x, _) = left, right
    radius = (right_x - left_x) / 2
    centre_x = left_x + radius
    # A transcription may place the ends the other way round, which is the same circle drawn backwards.
    steps = 2 * int(np.ceil(2 * np.pi * abs(radius) * canvas)) + 1
    for step, angle in enumerate(np.linspace(0, 2 * np.pi, steps)):
        if not dashed or (step // 8) % 2 == 0:
            _dot(image, centre_x + radius * np.cos(angle), y + radius * np.sin(angle), colour)


def _bend(image, start, mid, end, colour, dashed=False):
    """An arc, from start through mid to end along the circle the three lie on.

    Three points in a line lie on no circle, so they are drawn as the segments between them.
    """
    canvas = image.shape[0]
    (ax, ay), (bx, by), (cx, cy) = start, mid, end
    twice_area = 2 * (ax * (by - cy) + bx * (cy - ay) + cx * (ay - by))
    if abs(twice_area) < 1e-9:
        _segment(image, start, mid, colour, dashed)
        _segment(image, mid, end, colour, dashed)
        return
    squared = lambda x, y: x * x + y * y
    centre_x = (squared(ax, ay) * (by - cy) + squared(bx, by) * (cy - ay) + squared(cx, cy) * (ay - by)) / twice_area
    centre_y = (squared(ax, ay) * (cx - bx) + squared(bx, by) * (ax - cx) + squared(cx, cy) * (bx - ax)) / twice_area
    radius = np.hypot(ax - centre_x, ay - centre_y)
    angle = lambda x, y: np.arctan2(y - centre_y, x - centre_x)
    # Swept from start to end whichever way round passes the middle.
    begin, through, finish = angle(ax, ay), angle(bx, by), angle(cx, cy)
    turn = lambda to: (to - begin) % (2 * np.pi)
    sweep = turn(finish) if turn(through) <= turn(finish) else turn(finish) - 2 * np.pi
    steps = 2 * int(np.ceil(abs(sweep) * radius * canvas)) + 1
    for step, at in enumerate(np.linspace(begin, begin + sweep, steps)):
        if not dashed or (step // 8) % 2 == 0:
            _dot(image, centre_x + radius * np.cos(at), centre_y + radius * np.sin(at), colour)


def _dot(image, x, y, colour, radius=0):
    """One normalized point, as the pixels within `radius` of where it falls."""
    canvas = image.shape[0]
    at_x, at_y = int(round(x * canvas)), int(round(y * canvas))
    within = lambda at: slice(max(at - radius, 0), min(at + radius + 1, canvas))
    image[within(at_x), within(at_y)] = colour
