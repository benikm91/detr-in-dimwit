"""What a floor plan's record stands for, as a picture.

Every node is drawn as the segment between its two points, in the colour of its
class: the walls first, and the doors, outside doors and windows on top of the
walls they stand in. Every relationship is a dashed connector between the
middles of the two nodes it relates. The result is drawn over a drawing, so that
a record can be compared pixel by pixel with the plan it was read from.
"""

import numpy as np

#: Colours the classes and relationships are drawn in, as ``(red, green, blue)``, by their names in the record.
NODES = {
    "outside wall": (20, 60, 190),
    "inside wall": (20, 150, 210),
    "door": (200, 40, 60),
    "outside door": (230, 140, 20),
    "window": (20, 160, 110),
}
RELATIONSHIPS = {
    "joined": (120, 40, 160),
    "in": (170, 120, 200),
}


def render(drawing, node_class, start_x, start_y, end_x, end_y, relationship, subject, obj, classes, relationships):
    """``(width, height, 3)`` uint8 pixels of the record drawn over ``drawing``, a ``(width,
    height)`` grey level image. ``classes`` and ``relationships`` name the ids in order."""
    image = np.repeat(np.asarray(drawing, dtype=np.uint8)[:, :, None], 3, axis=2)
    start = list(zip(start_x, start_y))
    end = list(zip(end_x, end_y))
    middle = [((sx + ex) / 2, (sy + ey) / 2) for (sx, sy), (ex, ey) in zip(start, end)]

    for at, held in enumerate(relationship):
        colour = RELATIONSHIPS.get(relationships[held])
        if colour is not None:
            _segment(image, middle[subject[at]], middle[obj[at]], colour, dashed=True)

    walls_first = sorted(range(len(node_class)), key=lambda at: not classes[node_class[at]].endswith("wall"))
    for at in walls_first:
        colour = NODES.get(classes[node_class[at]])
        if colour is not None:
            _segment(image, start[at], end[at], colour, radius=1)

    return image


def _segment(image, start, end, colour, dashed=False, radius=0):
    """A straight run of pixels from one normalized point to another.

    Two steps per pixel it spans, so that a slope never leaves a gap between them.
    """
    canvas = image.shape[0]
    (x0, y0), (x1, y1) = start, end
    steps = 2 * int(np.ceil(max(abs(x1 - x0), abs(y1 - y0)) * canvas)) + 1
    for step, (x, y) in enumerate(zip(np.linspace(x0, x1, steps), np.linspace(y0, y1, steps))):
        if not dashed or (step // 8) % 2 == 0:
            _dot(image, x, y, colour, radius)


def _dot(image, x, y, colour, radius=0):
    """One normalized point, as the pixels within `radius` of where it falls."""
    canvas = image.shape[0]
    at_x, at_y = int(round(x * canvas)), int(round(y * canvas))
    within = lambda at: slice(max(at - radius, 0), min(at + radius + 1, canvas))
    image[within(at_x), within(at_y)] = colour
