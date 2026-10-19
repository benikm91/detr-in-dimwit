# Draws the same test plans as every run given transcribed them, a grid per run: the target in grey,
# its nodes no one wrote down in orange, the written nodes green where right and red where not. Walls
# are lines, outside walls thicker; a door or window lies on its wall's line, so it is marked by its
# two jambs, a tick across the line at either end: doors with short ticks, windows with long ones.
#
#   uv run --with matplotlib python draw_fp_transcripts.py <out folder> <run folder>... [--disagree]
#
# With --disagree, only plans that some of the runs get exactly right and others do not.
import json
import math
import random
import sys
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt

matplotlib.use("Agg")

STEP = 100_000
TOLERANCE = "4"
SIDE = 512
PLANS = 36

WALLS = {"outside wall", "inside wall"}

# The length of a jamb's tick, in points: how a door is told from a window.
JAMB = {"door": 6, "outside door": 6, "window": 11}


def draw(ax, node, colour, width):
    (x0, y0), (x1, y1) = node["points"]
    if node["class"] in WALLS:
        ax.plot([x0, x1], [y0, y1], color=colour, linewidth=width * (1.5 if node["class"] == "outside wall" else 1), solid_capstyle="butt")
    else:
        across = "|" if y0 == y1 else "_"
        ax.plot([x0, x1], [y0, y1], color=colour, linewidth=width, marker=across, markersize=JAMB[node["class"]], markeredgewidth=width)


def exact(plan):
    found = plan["matched"][TOLERANCE]
    return all(at is not None for at in found) and len(found) == len(plan["target"]["nodes"])


out, *folders = [arg for arg in sys.argv[1:] if arg != "--disagree"]
runs = {Path(folder).name: [json.loads(line) for line in open(next(Path(folder).glob(f"*-test-{STEP}.jsonl")))] for folder in folders}
candidates = [at for at in range(len(next(iter(runs.values()))))
              if "--disagree" not in sys.argv or len({exact(plans[at]) for plans in runs.values()}) > 1]
picked = sorted(random.Random(0).sample(candidates, min(PLANS, len(candidates))))

for name, plans in runs.items():
    side = math.ceil(math.sqrt(len(picked)))
    fig, axes = plt.subplots(side, side, figsize=(3.5 * side, 3.5 * side))
    for ax, at in zip(axes.flat, picked):
        plan = plans[at]
        target, written = plan["target"]["nodes"], plan["predicted"]["nodes"]
        found = plan["matched"][TOLERANCE]
        for index, node in enumerate(target):
            draw(ax, node, "#bdbdbd" if index in found else "#f4a261", 4)
        for node, match in zip(written, found):
            draw(ax, node, "#1a9850" if match is not None else "#d73027", 1.3)
        ax.set_title(f"#{at}: {len(target)} target, {len(written)} written", fontsize=9,
                     color="#1a9850" if exact(plan) else "#d73027")
        ax.set_xlim(0, SIDE)
        ax.set_ylim(SIDE, 0)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
    for ax in axes.flat[len(picked):]:
        ax.axis("off")
    fig.suptitle(f"{name}: {sum(exact(plans[at]) for at in picked)} of {len(picked)} exactly right", x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    fig.savefig(Path(out) / f"{name}-test.png", dpi=80)
    print(Path(out) / f"{name}-test.png")
