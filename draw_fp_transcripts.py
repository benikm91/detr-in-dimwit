# Draws the same test plans as every run given transcribed them, a grid per run: each plan as the
# drawing the model read, and beside it what the model wrote down, coloured by class. Walls are bars,
# a door or window a bar of its own colour on its wall's line. Furniture is a box with a dot on the
# side it faces. Rooms are not drawn: they are counted in the title, as found of the target's. The
# title is green when the plan is exactly right.
#
#   uv run --with matplotlib --with huggingface_hub python draw_fp_transcripts.py <out folder> <run folder>... [--disagree]
#
# Each run is drawn at its last scored checkpoint. With --disagree, only plans that some of the runs
# get exactly right and others do not. The drawings are read from the Hugging Face cache.
import json
import math
import random
import sys
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from huggingface_hub import hf_hub_download
from matplotlib.lines import Line2D

matplotlib.use("Agg")

TOLERANCE = "4"
PLANS = 36

REPOSITORY = {"synthetic-floorplans": "benikm91/synthetic-floorplans-for-transcription",
              "synthetic-floorplans-furnished": "benikm91/synthetic-floorplans-with-furnitures-for-transcription"}

COLOUR = {"outside wall": "#2b4d86", "inside wall": "#6a9fd4", "window": "#4a9b55", "door": "#e8956a",
          "outside door": "#c62f2f"}
FURNITURE = "#8c6d46"


def draw(ax, node):
    (x0, y0), (x1, y1) = node["points"]
    if node["class"] in COLOUR:
        ax.plot([x0, x1], [y0, y1], color=COLOUR[node["class"]], linewidth=4, solid_capstyle="butt",
                zorder=2 if node["class"].endswith("wall") else 3)
    else:
        # The back-left corner, then the front-right one: the box faces down or up when the two
        # differ in y the same way as in x, and left or right otherwise.
        facesUpOrDown = (x1 - x0) * (y1 - y0) > 0
        front = ((x0 + x1) / 2, y1) if facesUpOrDown else (x1, (y0 + y1) / 2)
        ax.plot([x0, x1, x1, x0, x0], [y0, y0, y1, y1, y0], color=FURNITURE, linewidth=1)
        ax.plot(*front, "o", color=FURNITURE, markersize=3)


def placed(node):
    return "points" in node


def exact(plan):
    found = plan["matched"][TOLERANCE]
    return all(at is not None for at in found) and len(found) == len(plan["target"]["nodes"])


def lastScored(folder):
    scored = list(folder.glob("*-test-*.jsonl"))
    if not scored:
        sys.exit(f"no test transcripts (*-test-<step>.jsonl) in {folder}")
    return max(scored, key=lambda path: int(path.stem.rsplit("-", 1)[1]))


def blank(ax, side):
    ax.set_xlim(0, side)
    ax.set_ylim(side, 0)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])


out, *folders = [arg for arg in sys.argv[1:] if arg != "--disagree"]
runs = {}
for folder in map(Path, folders):
    transcripts = lastScored(folder)
    corpus = json.loads(next(folder.glob("*.environment.json")).read_text())["corpus"]
    drawings = np.load(hf_hub_download(REPOSITORY[corpus], "test_images.npy", repo_type="dataset"), mmap_mode="r")
    runs[folder.name] = (transcripts.stem.rsplit("-", 1)[1], drawings, [json.loads(line) for line in open(transcripts)])

candidates = [at for at in range(len(next(iter(runs.values()))[2]))
              if "--disagree" not in sys.argv or len({exact(plans[at]) for _, _, plans in runs.values()}) > 1]
picked = sorted(random.Random(0).sample(candidates, min(PLANS, len(candidates))))

legend = [Line2D([], [], color=colour, linewidth=4, label=name) for name, colour in COLOUR.items()] + \
         [Line2D([], [], color=FURNITURE, linewidth=1, marker="o", markersize=3, label="furniture")]

for name, (step, drawings, plans) in runs.items():
    side = drawings.shape[1]
    pairsPerRow = math.ceil(math.sqrt(len(picked)))
    rows = math.ceil(len(picked) / pairsPerRow)
    fig, axes = plt.subplots(rows, 2 * pairsPerRow, figsize=(8 * pairsPerRow, 4 * rows), squeeze=False)
    for (read, written), at in zip(axes.reshape(-1, 2), picked):
        plan = plans[at]
        target, nodes = plan["target"]["nodes"], plan["predicted"]["nodes"]
        found = plan["matched"][TOLERANCE]
        read.imshow(drawings[plan["plan"]], cmap="gray", vmin=0, vmax=255, extent=(0, side, side, 0))
        for node in nodes:
            if placed(node):
                draw(written, node)
        rooms = sum(not placed(node) for node in target)
        roomsFound = sum(match is not None and not placed(target[match]) for match in found)
        roomsWritten = sum(not placed(node) for node in nodes)
        heldRooms = f", rooms {roomsFound} of {rooms} found, {roomsWritten} written" if rooms or roomsWritten else ""
        read.set_title(f"#{at}", fontsize=9)
        written.set_title(f"{len(target)} target, {len(nodes)} written{heldRooms}", fontsize=9,
                          color="#1a9850" if exact(plan) else "#d73027")
        blank(read, side)
        blank(written, side)
    for ax in axes.flat[2 * len(picked):]:
        ax.axis("off")
    fig.suptitle(f"{name} at {int(step) // 1000}k: {sum(exact(plans[at]) for at in picked)} of {len(picked)} exactly right", x=0.01, ha="left")
    fig.legend(handles=legend, loc="upper right", ncol=len(legend), frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    fig.savefig(Path(out) / f"{name}-test.png", dpi=80)
    print(Path(out) / f"{name}-test.png")
