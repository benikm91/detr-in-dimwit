# %% What to look at
# How the models go wrong on the validation split of sketch-xl: what each wrote down for every
# drawing, as the eval left it in <model>-sketch-xl-s-deep-<step>.jsonl beside the scores.
# Run cell by cell in an interactive window, or the whole file for every plot.
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

RUNS = Path("results/runs")

TRANSCRIPTS = {
    "d2s 6e-4": RUNS / "xl-512-lr6e-4-d2s-b098-wd05/d2s-sketch-xl-s-deep-125000.jsonl",
    "detr q16 6e-4": RUNS / "xl-512-lr6e-4-detr-q16-b098-wd05/detr-sketch-xl-s-deep-125000.jsonl",
    "detr q16 3e-4": RUNS / "xl-512-lr3e-4-detr-q16-b098-wd05/detr-sketch-xl-s-deep-125000.jsonl",
}

# The tolerance a node is matched at, in pixels, and a looser one: a node matched only at the looser
# one is in the right place but not quite.
TOLERANCE, LOOSE = "4", "8"


# %% Load: one row per drawing
def drawings(path):
    rows = []
    with open(path) as lines:
        for line in lines:
            drawing = json.loads(line)
            target, predicted = drawing["target"]["nodes"], drawing["predicted"]["nodes"]
            matched = drawing["matched"][TOLERANCE]
            right = [at is not None for at in matched]
            # How many nodes the model wrote down right before its first wrong one, in the order it wrote them.
            rightFirst = next((at for at, isRight in enumerate(right) if not isRight), len(right))
            rows.append({
                "drawing": drawing["drawing"],
                "nodes": len(target),
                "written": len(predicted),
                "right": sum(right),
                "nearly": sum(at is not None for at in drawing["matched"][LOOSE]),
                "rightFirst": rightFirst,
                "rightAfter": sum(right[rightFirst:]),
            })
    rows = pd.DataFrame(rows).set_index("drawing")
    rows["missed"] = rows.nodes - rows.right
    rows["extra"] = rows.written - rows.right
    # A node written in the wrong place is both a miss and an extra, but one node to put right.
    rows["errors"] = np.maximum(rows.missed, rows.extra)
    rows["exact"] = rows.errors == 0
    return rows


runs = {name: drawings(path) for name, path in TRANSCRIPTS.items()}
assert len({len(rows) for rows in runs.values()}) == 1, "the runs were scored on different drawings"

plt.rcParams.update({
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.color": "#e6e6e3",
    "axes.edgecolor": "#c3c2b7",
    "lines.linewidth": 2,
    "legend.frameon": False,
})
COLOURS = dict(zip(TRANSCRIPTS, ["#d85a30", "#185fa5", "#85b7eb"]))


# %% Node scores pooled over every node, and averaged over drawings
def summary(rows):
    return {
        "exact": rows.exact.mean() * 100,
        "≤ 1 error": (rows.errors <= 1).mean() * 100,
        "≤ 2 errors": (rows.errors <= 2).mean() * 100,
        "≥ 4 errors": (rows.errors >= 4).mean() * 100,
        "found, pooled": rows.right.sum() / rows.nodes.sum() * 100,
        "right, pooled": rows.right.sum() / rows.written.sum() * 100,
        "found, per drawing": (rows.right / rows.nodes).mean() * 100,
        "right, per drawing": (rows.right / rows.written.clip(lower=1)).mean() * 100,
        "errors per drawing": rows.errors.mean(),
        "errors per wrong drawing": rows.errors[~rows.exact].mean(),
        "missed": rows.missed.sum() / rows.nodes.sum() * 100,
        "extra": rows.extra.sum() / rows.nodes.sum() * 100,
        "off by 4–8 px": (rows.nearly - rows.right).sum() / rows.nodes.sum() * 100,
        "too few written": (rows.written < rows.nodes).mean() * 100,
        "too many written": (rows.written > rows.nodes).mean() * 100,
    }


print(f"at {TOLERANCE} px, % of drawings or of target nodes\n")
print(pd.DataFrame({name: summary(rows) for name, rows in runs.items()}).round(1).to_string())


# %% How many nodes a drawing gets wrong: missed plus extra
fig, ax = plt.subplots(figsize=(10, 5))
bins = np.arange(0, 8)
width = 0.8 / len(runs)
for offset, (name, rows) in enumerate(runs.items()):
    share = [(rows.errors.clip(upper=bins[-1]) == at).mean() * 100 for at in bins]
    ax.bar(bins + offset * width - 0.4 + width / 2, share, width, color=COLOURS[name], label=name)
ax.set_xticks(bins, [str(at) for at in bins[:-1]] + [f"{bins[-1]}+"])
ax.set_xlabel(f"nodes to put right in the drawing (at {TOLERANCE} px)")
ax.set_ylabel("% of drawings")
ax.legend()
fig.tight_layout()
plt.show()


# %% Whole drawings right, by how many nodes the drawing holds
fig, axes = plt.subplots(1, 2, figsize=(16, 5))
for name, rows in runs.items():
    bySize = rows.groupby("nodes")
    axes[0].plot(bySize.exact.mean().index, bySize.exact.mean() * 100, color=COLOURS[name], marker="o", label=name)
    axes[1].plot(bySize.errors.mean().index, bySize.errors.mean(), color=COLOURS[name], marker="o", label=name)
counts = next(iter(runs.values())).nodes.value_counts().sort_index()
for nodes, count in counts.items():
    axes[0].annotate(f"{count}", (nodes, 2), ha="center", fontsize=7, color="#888780")
axes[0].set_title("drawings exactly right (grey: how many drawings)", loc="left")
axes[0].set_ylabel("%")
axes[0].set_ylim(0, 100)
axes[1].set_title("nodes to put right per drawing", loc="left")
for ax in axes:
    ax.set_xlabel("nodes in the drawing")
axes[0].legend()
fig.tight_layout()
plt.show()


# %% d2s writes in order: does it recover after its first mistake?
d2s = runs["d2s 6e-4"]
wrong = d2s[~d2s.exact]
writtenAfter = wrong.written - wrong.rightFirst - 1
recovered = wrong.rightAfter / writtenAfter.where(writtenAfter > 0)
print(f"d2s, the {len(wrong)} drawings it gets wrong:")
print(f"  first mistake in its own order: at node {wrong.rightFirst.mean():.1f} of {wrong.written.mean():.1f} written, on average")
print(f"  first mistake is the end (it stopped too early, with every node written right): {(wrong.rightFirst == wrong.written).mean() * 100:.1f}%")
print(f"  of the nodes written after the first mistake, right: {wrong.rightAfter.sum() / writtenAfter.clip(lower=0).sum() * 100:.1f}%")

fig, ax = plt.subplots(figsize=(10, 5))
relative = (wrong.rightFirst / wrong.nodes).clip(upper=1)
ax.hist(relative, bins=np.linspace(0, 1, 11), color=COLOURS["d2s 6e-4"], weights=np.full(len(wrong), 100 / len(wrong)))
ax.set_xlabel("how far into the drawing the first mistake comes (nodes right before it / nodes in the drawing)")
ax.set_ylabel("% of the drawings d2s gets wrong")
fig.tight_layout()
plt.show()


# %% The same drawings: which model gets which right
names = list(runs)
for other in names[1:]:
    both = pd.crosstab(runs[names[0]].exact, runs[other].exact, normalize=True) * 100
    both.index = [f"{names[0]} wrong", f"{names[0]} right"]
    both.columns = [f"{other} wrong", f"{other} right"]
    print(f"\n% of drawings\n{both.round(1).to_string()}")


# %% Duplicates removed afterwards: how much of DETR's gap is two queries writing the same node
def isAt(node, other, pixels):
    """Two nodes of the same class whose points are all within `pixels` of one another, in order."""
    return node["class"] == other["class"] and all(
        abs(a - b) <= pixels for at, by in zip(node["points"], other["points"]) for a, b in zip(at, by)
    )


def matchedCount(target, predicted, pixels):
    """How many target nodes are found, matched one to one in target order as the eval does."""
    taken = set()
    for node in target:
        slot = next((at for at, found in enumerate(predicted) if at not in taken and isAt(node, found, pixels)), None)
        if slot is not None:
            taken.add(slot)
    return len(taken)


def deduplicated(predicted, pixels):
    """Every node but those that repeat one written before them."""
    return [node for at, node in enumerate(predicted) if not any(isAt(node, kept, pixels) for kept in predicted[:at])]


def exactWithout(path, duplicateWithin):
    exact, extraOnly = 0, 0
    with open(path) as lines:
        for line in lines:
            drawing = json.loads(line)
            target = drawing["target"]["nodes"]
            predicted = deduplicated(drawing["predicted"]["nodes"], duplicateWithin)
            right = matchedCount(target, predicted, int(TOLERANCE))
            exact += right == len(target) == len(predicted)
            extraOnly += right == len(target) < len(predicted)
    return exact, extraOnly


rows = []
for name, path in TRANSCRIPTS.items():
    total = len(runs[name])
    for within in [None, 2, 4, 8]:
        exact, extraOnly = exactWithout(path, -1 if within is None else within)
        rows.append({"run": name, "duplicates within": "kept" if within is None else f"{within} px",
                     "exact": exact / total * 100, "every node found, plus extras": extraOnly / total * 100})
    # The most removing nodes afterwards could give: every extra node gone, and no missed one found.
    rows.append({"run": name, "duplicates within": "every extra node", "exact": (runs[name].missed == 0).mean() * 100,
                 "every node found, plus extras": 0.0})
print(pd.DataFrame(rows).pivot(index="duplicates within", columns="run", values="exact").round(1).to_string())
print()
print(pd.DataFrame(rows).pivot(index="duplicates within", columns="run", values="every node found, plus extras").round(1).to_string())
