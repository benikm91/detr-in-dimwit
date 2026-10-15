# Draws the same test sketches as every run given transcribed them, a grid per run: the target in
# grey, its nodes no one wrote down in orange, the written nodes green where right and red where not,
# construction nodes dashed.
#
#   uv run --with matplotlib --with numpy python draw_transcripts.py <out folder> <run folder>... [--disagree]
#
# With --disagree, only sketches that some of the runs get exactly right and others do not.
import json
import math
import random
import sys
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.use("Agg")

STEP = 270_000
TOLERANCE = "4"
CANVAS = 128
SKETCHES = 36


def traced(node, steps=64):
    """The points a node is drawn through; an arc's are its start, end and a point it passes through."""
    p = np.array(node["points"], dtype=float)
    if node["class"] == "point":
        return p
    if node["class"] == "line":
        return p[0] + np.linspace(0, 1, steps)[:, None] * (p[1] - p[0])
    if node["class"] == "circle":
        centre, radius = (p[0] + p[1]) / 2, np.linalg.norm(p[1] - p[0]) / 2
        angles = np.linspace(0, 2 * math.pi, steps)
        return centre + radius * np.stack([np.cos(angles), np.sin(angles)], 1)
    start, through, end = p[0], p[2], p[1]
    twice = 2 * (start[0] * (through[1] - end[1]) + through[0] * (end[1] - start[1]) + end[0] * (start[1] - through[1]))
    if abs(twice) < 1e-9:
        return np.stack([start, through, end])
    square = lambda q: q @ q
    centre = np.array([
        square(start) * (through[1] - end[1]) + square(through) * (end[1] - start[1]) + square(end) * (start[1] - through[1]),
        square(start) * (end[0] - through[0]) + square(through) * (start[0] - end[0]) + square(end) * (through[0] - start[0]),
    ]) / twice
    angle = lambda q: math.atan2(q[1] - centre[1], q[0] - centre[0])
    turn = lambda to: (angle(to) - angle(start)) % (2 * math.pi)
    sweep = turn(end) if turn(through) <= turn(end) else turn(end) - 2 * math.pi
    angles = angle(start) + np.linspace(0, sweep, steps)
    return centre + np.linalg.norm(start - centre) * np.stack([np.cos(angles), np.sin(angles)], 1)


def exact(drawing):
    found = drawing["matched"][TOLERANCE]
    return all(at is not None for at in found) and len(found) == len(drawing["target"]["nodes"])


out, *folders = [arg for arg in sys.argv[1:] if arg != "--disagree"]
runs = {Path(folder).name: [json.loads(line) for line in open(next(Path(folder).glob(f"*-test-{STEP}.jsonl")))] for folder in folders}
candidates = [at for at in range(len(next(iter(runs.values()))))
              if "--disagree" not in sys.argv or len({exact(drawings[at]) for drawings in runs.values()}) > 1]
picked = sorted(random.Random(0).sample(candidates, min(SKETCHES, len(candidates))))

for name, drawings in runs.items():
    side = math.ceil(math.sqrt(len(picked)))
    fig, axes = plt.subplots(side, side, figsize=(3 * side, 3 * side))
    for ax, at in zip(axes.flat, picked):
        drawing = drawings[at]
        target, written = drawing["target"]["nodes"], drawing["predicted"]["nodes"]
        found = drawing["matched"][TOLERANCE]
        for index, node in enumerate(target):
            line = traced(node)
            ax.plot(line[:, 0], line[:, 1], color="#bdbdbd" if index in found else "#f4a261", linewidth=5,
                    marker="o" if node["class"] == "point" else None, markersize=7)
        for node, match in zip(written, found):
            line = traced(node)
            ax.plot(line[:, 0], line[:, 1], color="#1a9850" if match is not None else "#d73027", linewidth=1.3,
                    linestyle="--" if node["construction"] else "-", marker="o" if node["class"] == "point" else None, markersize=3)
        ax.set_title(f"#{at}: {len(target)} target, {len(written)} written", fontsize=9,
                     color="#1a9850" if exact(drawing) else "#d73027")
        ax.set_xlim(0, CANVAS)
        ax.set_ylim(CANVAS, 0)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
    for ax in axes.flat[len(picked):]:
        ax.axis("off")
    fig.suptitle(f"{name}: {sum(exact(drawings[at]) for at in picked)} of {len(picked)} exactly right", x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    fig.savefig(Path(out) / f"{name}-test.png", dpi=80)
    print(Path(out) / f"{name}-test.png")
