# %% What to look at
# How much of sketch-xl a model cannot get exactly right from the drawing alone: sketches whose
# record holds something the picture does not show — a node drawn twice, a line split where nothing
# shows the split, an arc lying on a circle — against what the two best runs got right.
# Run cell by cell in an interactive window, or the whole file for every plot.
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

RUNS = Path("results/runs")

TRANSCRIPTS = {
    "d2s": RUNS / "xl-512-lr6e-4-d2s-b098-wd05/d2s-sketch-xl-s-deep-125000.jsonl",
    "detr": RUNS / "xl-512-lr6e-4-detr-q16-b098-wd05/detr-sketch-xl-s-deep-125000.jsonl",
}

# The tolerance the runs are compared at, in pixels.
TOLERANCE = "4"

# How close two strokes must come, in pixels, to be one stroke in the drawing.
SAME_STROKE = 1.0

# Smaller than this, in pixels, a node is too small for its shape to be seen.
TOO_SMALL = 1.5


# %% The geometry of a node, in pixels
def circle_of(node):
    """The centre and radius of a circle or an arc, or None for an arc too straight to bend."""
    points = np.array(node["points"], dtype=float)
    if node["class"] == "circle":
        return (points[0] + points[1]) / 2, np.linalg.norm(points[1] - points[0]) / 2
    a, b, c = points[0], points[2], points[1]  # start, middle, end: a record holds an arc as start, end, middle
    twice = 2 * (a[0] * (b[1] - c[1]) + b[0] * (c[1] - a[1]) + c[0] * (a[1] - b[1]))
    if abs(twice) < 1e-9:
        return None
    squared = lambda p: p @ p
    centre = np.array([
        squared(a) * (b[1] - c[1]) + squared(b) * (c[1] - a[1]) + squared(c) * (a[1] - b[1]),
        squared(a) * (c[0] - b[0]) + squared(b) * (a[0] - c[0]) + squared(c) * (b[0] - a[0]),
    ]) / twice
    return centre, np.linalg.norm(a - centre)


def traced(node, steps=64):
    """Points along a node's stroke."""
    points = np.array(node["points"], dtype=float)
    if node["class"] == "line":
        return points[0] + np.linspace(0, 1, steps)[:, None] * (points[1] - points[0])
    held = circle_of(node)
    if held is None:
        return points
    centre, radius = held
    if node["class"] == "circle":
        angles = np.linspace(0, 2 * math.pi, steps)
    else:
        angle = lambda p: math.atan2(p[1] - centre[1], p[0] - centre[0])
        begin, through, finish = angle(points[0]), angle(points[2]), angle(points[1])
        turn = lambda to: (to - begin) % (2 * math.pi)
        sweep = turn(finish) if turn(through) <= turn(finish) else turn(finish) - 2 * math.pi
        angles = begin + np.linspace(0, sweep, steps)
    return centre + radius * np.stack([np.cos(angles), np.sin(angles)], axis=1)


def size_of(node):
    points = np.array(node["points"], dtype=float)
    if node["class"] == "line":
        return np.linalg.norm(points[1] - points[0])
    held = circle_of(node)
    return math.inf if held is None else held[1]


# %% What a record holds that its drawing does not show
def drawn_twice(a, b):
    """Two nodes of one class whose points all coincide, a line's in either order."""
    if a["class"] != b["class"]:
        return False
    pa, pb = np.array(a["points"]), np.array(b["points"])
    alike = lambda p, q: np.all(np.linalg.norm(p - q, axis=1) <= SAME_STROKE)
    return alike(pa, pb) or (a["class"] == "line" and alike(pa, pb[::-1]))


def one_line_split(a, b):
    """Two lines on one straight stroke, touching or overlapping: drawn, they are one line."""
    if a["class"] != "line" or b["class"] != "line":
        return False
    (a0, a1), (b0, b1) = np.array(a["points"]), np.array(b["points"])
    length = np.linalg.norm(a1 - a0)
    if length == 0:
        return False
    along = (a1 - a0) / length
    off = lambda p: abs(along[0] * (p - a0)[1] - along[1] * (p - a0)[0])
    if off(b0) > SAME_STROKE or off(b1) > SAME_STROKE:
        return False
    span = sorted([(b0 - a0) @ along, (b1 - a0) @ along])
    return span[0] <= length + SAME_STROKE and span[1] >= -SAME_STROKE


def one_curve_split(a, b):
    """A circle or an arc lying on the stroke of another: an arc along a circle, or two arcs of one
    circle that touch or overlap."""
    if a["class"] not in ("circle", "arc") or b["class"] not in ("circle", "arc"):
        return False
    ca, cb = circle_of(a), circle_of(b)
    if ca is None or cb is None:
        return False
    if np.linalg.norm(ca[0] - cb[0]) > SAME_STROKE or abs(ca[1] - cb[1]) > SAME_STROKE:
        return False
    if "circle" in (a["class"], b["class"]):
        return True
    ends = np.array([a["points"][0], a["points"][1], b["points"][0], b["points"][1]], dtype=float)
    reach = lambda points, stroke: np.min(np.linalg.norm(points[:, None] - stroke[None], axis=2), axis=1) <= SAME_STROKE
    return reach(ends[2:], traced(a)).any() or reach(ends[:2], traced(b)).any()


PAIRED = {"drawn twice": drawn_twice, "one line split": one_line_split, "one curve split": one_curve_split}


def hidden(nodes):
    """Which of the rules a record breaks: pairs its drawing cannot tell apart, and nodes too small to see."""
    found = {rule: False for rule in [*PAIRED, "too small to see"]}
    for i, a in enumerate(nodes):
        found["too small to see"] |= size_of(a) < TOO_SMALL
        for b in nodes[i + 1:]:
            for rule, holds in PAIRED.items():
                found[rule] |= holds(a, b)
    return found


# %% Load: per drawing, what its record hides and whether each run wrote it exactly
def exact(drawing):
    matched = drawing["matched"][TOLERANCE]
    return all(at is not None for at in matched) and len(matched) == len(drawing["target"]["nodes"])


with open(TRANSCRIPTS["d2s"]) as lines:
    targets = [json.loads(line)["target"]["nodes"] for line in lines]
rows = pd.DataFrame([hidden(nodes) for nodes in targets])
rows["ambiguous"] = rows.any(axis=1)
rows["nodes"] = [len(nodes) for nodes in targets]
for name, path in TRANSCRIPTS.items():
    with open(path) as lines:
        rows[name] = [exact(json.loads(line)) for line in lines]

print(f"{len(rows)} validation sketches; how many hold what their drawing does not show:")
for rule in [*PAIRED, "too small to see", "ambiguous"]:
    print(f"  {rule:18s} {rows[rule].mean() * 100:5.1f}%")


# %% Exactly right, where the drawing shows everything and where it does not
def share(held):
    return pd.Series({
        "sketches": f"{len(held) / len(rows) * 100:.1f}%",
        "d2s exact": round(held.d2s.mean() * 100, 1),
        "detr exact": round(held.detr.mean() * 100, 1),
        "both exact": round((held.d2s & held.detr).mean() * 100, 1),
        "neither": round((~held.d2s & ~held.detr).mean() * 100, 1),
    })


table = pd.DataFrame({
    "all": share(rows),
    "unambiguous": share(rows[~rows.ambiguous]),
    "ambiguous": share(rows[rows.ambiguous]),
    **{rule: share(rows[rows[rule]]) for rule in [*PAIRED, "too small to see"]},
}).T
print(f"\nat {TOLERANCE} px, % of the sketches of each row\n{table.to_string()}")

neither = rows[~rows.d2s & ~rows.detr]
print(f"\nof the sketches neither run gets exactly right, {neither.ambiguous.mean() * 100:.1f}% are ambiguous")

unambiguous = rows[~rows.ambiguous]
print("\nunambiguous sketches, by how many nodes they hold: % exactly right")
print(unambiguous.groupby(pd.cut(unambiguous.nodes, [0, 6, 8, 10, 12, 16]), observed=True)[["d2s", "detr"]].mean().mul(100).round(1).to_string())


# %% A sample of ambiguous sketches, with the nodes that make them so in red, to check the rules by eye
def offending(nodes):
    marked = set()
    for i, a in enumerate(nodes):
        if size_of(a) < TOO_SMALL:
            marked.add(i)
        for j in range(i + 1, len(nodes)):
            if any(holds(a, nodes[j]) for holds in PAIRED.values()):
                marked |= {i, j}
    return marked


sample = rows[rows.ambiguous].sample(16, random_state=0).index
fig, axes = plt.subplots(4, 4, figsize=(14, 14))
for ax, at in zip(axes.flat, sample):
    nodes, marked = targets[at], offending(targets[at])
    for k, node in enumerate(nodes):
        stroke = traced(node)
        ax.plot(stroke[:, 0], stroke[:, 1], color="#c0392b" if k in marked else "#2c2c2a",
                linewidth=2.5 if k in marked else 1, alpha=0.7 if k in marked else 1)
    rules = [rule for rule in [*PAIRED, "too small to see"] if rows.at[at, rule]]
    ax.set_title(f"{at}: {', '.join(rules)}\nd2s {'✓' if rows.at[at, 'd2s'] else '✗'}  detr {'✓' if rows.at[at, 'detr'] else '✗'}", fontsize=8)
    ax.set_xlim(0, 256)
    ax.set_ylim(256, 0)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
fig.tight_layout()
plt.show()


# %% Sketches exactly right by how many primitives they hold, with d2s corrected once beside them
corrections = pd.read_csv(RUNS / "xl-512-lr6e-4-d2s-b098-wd05/d2s-sketch-xl-s-deep-125000-corrections.csv").set_index("drawing")
rows["d2s, one correction"] = corrections.helps.values <= 1

LINES = {
    "d2s": dict(color="#d85a30", linestyle="-"),
    "detr": dict(color="#185fa5", linestyle="-"),
    "d2s, one correction": dict(color="#d85a30", linestyle="--"),
}
fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharey=True)
for ax, (title, held) in zip(axes, [("all sketches", rows), ("unambiguous sketches", rows[~rows.ambiguous])]):
    by_size = held.groupby("nodes")
    for name, style in LINES.items():
        ax.plot(by_size[name].mean().index, by_size[name].mean() * 100, marker="o", markersize=4, label=name, **style)
    for nodes, count in by_size.size().items():
        ax.annotate(f"{count}", (nodes, 2), ha="center", fontsize=7, color="#888780")
    ax.set_title(f"{title} (grey: how many)", loc="left")
    ax.set_xlabel("primitives in the sketch")
    ax.set_xticks(range(held.nodes.min(), held.nodes.max() + 1))
    ax.set_ylim(0, 100)
axes[0].set_ylabel(f"% exactly right at {TOLERANCE} px")
axes[0].legend()
fig.suptitle("sketch-xl validation — d2s and DETR (16 queries), lr 6e-4, batch 512", x=0.01, ha="left")
fig.tight_layout()
plt.show()
