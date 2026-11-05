# Draws the same test plans as every run given transcribed them, a grid per run: the target in grey,
# its nodes no one wrote down in orange, the written nodes green where right and red where not. Walls
# are lines, outside walls thicker; a door or window lies on its wall's line, so it is marked by its
# two jambs, a tick across the line at either end: doors with short ticks, windows with long ones.
# Furniture is a box with a dot on the side it faces. Rooms are not drawn: they are counted in the
# title, as found of the target's.
#
#   uv run --with matplotlib python draw_fp_transcripts.py <out folder> <run folder>... [--disagree]
#
# Each run is drawn at its last scored checkpoint. With --disagree, only plans that some of the runs
# get exactly right and others do not.
import json
import math
import random
import sys
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt

matplotlib.use("Agg")

TOLERANCE = "4"
PLANS = 36

# The side the drawings of a corpus are stored at, which the points are given in.
SIDE = {"synthetic-floorplans": 512, "synthetic-floorplans-furnished": 256}

WALLS = {"outside wall", "inside wall"}

# The length of a jamb's tick, in points: how a door is told from a window.
JAMB = {"door": 6, "outside door": 6, "window": 11}


def draw(ax, node, colour, width):
    (x0, y0), (x1, y1) = node["points"]
    if node["class"] in WALLS:
        ax.plot([x0, x1], [y0, y1], color=colour, linewidth=width * (1.5 if node["class"] == "outside wall" else 1), solid_capstyle="butt")
    elif node["class"] in JAMB:
        across = "|" if y0 == y1 else "_"
        ax.plot([x0, x1], [y0, y1], color=colour, linewidth=width, marker=across, markersize=JAMB[node["class"]], markeredgewidth=width)
    else:
        # The back-left corner, then the front-right one: the box faces down or up when the two
        # differ in y the same way as in x, and left or right otherwise.
        facesUpOrDown = (x1 - x0) * (y1 - y0) > 0
        front = ((x0 + x1) / 2, y1) if facesUpOrDown else (x1, (y0 + y1) / 2)
        ax.plot([x0, x1, x1, x0, x0], [y0, y0, y1, y1, y0], color=colour, linewidth=width / 2)
        ax.plot(*front, "o", color=colour, markersize=width * 1.5)


def placed(node):
    return "points" in node


def exact(plan):
    found = plan["matched"][TOLERANCE]
    return all(at is not None for at in found) and len(found) == len(plan["target"]["nodes"])


def lastScored(folder):
    return max(folder.glob("*-test-*.jsonl"), key=lambda path: int(path.stem.rsplit("-", 1)[1]))


out, *folders = [arg for arg in sys.argv[1:] if arg != "--disagree"]
runs = {}
for folder in map(Path, folders):
    transcripts = lastScored(folder)
    corpus = json.loads(next(folder.glob("*.environment.json")).read_text())["corpus"]
    runs[folder.name] = (transcripts.stem.rsplit("-", 1)[1], SIDE[corpus], [json.loads(line) for line in open(transcripts)])

candidates = [at for at in range(len(next(iter(runs.values()))[2]))
              if "--disagree" not in sys.argv or len({exact(plans[at]) for _, _, plans in runs.values()}) > 1]
picked = sorted(random.Random(0).sample(candidates, min(PLANS, len(candidates))))

for name, (step, side, plans) in runs.items():
    columns = math.ceil(math.sqrt(len(picked)))
    fig, axes = plt.subplots(columns, columns, figsize=(5 * columns, 5 * columns))
    for ax, at in zip(axes.flat, picked):
        plan = plans[at]
        target, written = plan["target"]["nodes"], plan["predicted"]["nodes"]
        found = plan["matched"][TOLERANCE]
        for index, node in enumerate(target):
            if placed(node):
                draw(ax, node, "#bdbdbd" if index in found else "#f4a261", 4)
        for node, match in zip(written, found):
            if placed(node):
                draw(ax, node, "#1a9850" if match is not None else "#d73027", 1.3)
        rooms = sum(not placed(node) for node in target)
        roomsFound = sum(match is not None and not placed(target[match]) for match in found)
        roomsWritten = sum(not placed(node) for node in written)
        heldRooms = f", rooms {roomsFound} of {rooms} found, {roomsWritten} written" if rooms or roomsWritten else ""
        ax.set_title(f"#{at}: {len(target)} target, {len(written)} written{heldRooms}", fontsize=9,
                     color="#1a9850" if exact(plan) else "#d73027")
        ax.set_xlim(0, side)
        ax.set_ylim(side, 0)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
    for ax in axes.flat[len(picked):]:
        ax.axis("off")
    fig.suptitle(f"{name} at {int(step) // 1000}k: {sum(exact(plans[at]) for at in picked)} of {len(picked)} exactly right", x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    fig.savefig(Path(out) / f"{name}-test.png", dpi=80)
    print(Path(out) / f"{name}-test.png")
