# %% What to look at
# How the models go wrong on Vitruvion's test split: how it depends on the size of the sketch,
# what kind of mistake a missed or extra node is, and where in its own writing a model makes it.
# Read from the transcripts the test eval leaves beside the scores, every seed of a configuration.
# Run cell by cell in an interactive window, or the whole file for every plot.
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

RUNS = Path("results/runs")
STEP = 270_000

SEEDS = (0, 1, 42)
CONFIGURATIONS = {
    ("d2seq", "kept"): [f"vitruvion-d2seq-kept-seed{s}" for s in SEEDS],
    ("d2s", "free"): ["vitruvion-final-d2s", "vitruvion-final-d2s-seed0", "vitruvion-final-d2s-seed1"],
    ("d2s", "class"): ["vitruvion-final-d2po", "vitruvion-final-d2po-seed0", "vitruvion-final-d2po-seed1"],
    ("d2s", "class reversed"): [f"vitruvion-final-d2po-reversed-seed{s}" for s in SEEDS],
    ("DETR", "free"): ["vitruvion-final-detr-q16", "vitruvion-final-detr-q16-seed1", "vitruvion-final-detr-q16-seed42"],
    ("d2tok", "random"): [f"vitruvion-d2tok-random-seed{s}" for s in SEEDS],
    ("d2tok", "kept"): [f"vitruvion-d2tok-kept-seed{s}" for s in SEEDS],
    ("d2tok", "class"): [f"vitruvion-d2tok-class-seed{s}" for s in SEEDS],
    ("d2tok", "position"): [f"vitruvion-d2tok-position-seed{s}" for s in SEEDS],
}

# The tolerance a node is matched at, in pixels, and a looser one: a node matched only at the looser
# one is in the right place but not quite.
TOLERANCE, LOOSE = 4, 8

HUES = {"d2seq": "#e87ba4", "d2s": "#2a78d6", "DETR": "#eb6834", "d2tok": "#1baf7a"}
STYLES = {"random": "-", "free": "-", "class": "--", "class reversed": ":", "kept": "-.", "position": (0, (1, 3))}


# %% Load: one row per drawing and seed, its mistakes sorted into kinds
def isAt(node, other, pixels):
    return all(abs(a - b) <= pixels for at, by in zip(node["points"], other["points"]) for a, b in zip(at, by))


def drawings(path):
    rows = []
    with open(path) as lines:
        for line in lines:
            drawing = json.loads(line)
            target, written = drawing["target"]["nodes"], drawing["predicted"]["nodes"]
            found = drawing["matched"][str(TOLERANCE)]
            missed = [node for at, node in enumerate(target) if at not in set(found)]
            extra = [node for node, at in zip(written, found) if at is None]
            right = [node for node, at in zip(written, found) if at is not None]

            # A missed node explained by an extra one in its place: what the extra one got wrong.
            def kind(node):
                if any(other["class"] == node["class"] and isAt(node, other, TOLERANCE) for other in extra):
                    return "construction flag"
                if any(len(other["points"]) == len(node["points"]) and isAt(node, other, TOLERANCE) for other in extra):
                    return "class"
                if any(other["class"] == node["class"] and isAt(node, other, LOOSE) for other in extra):
                    return f"off by {TOLERANCE}–{LOOSE} px"
                return "not written"

            kinds = pd.Series([kind(node) for node in missed], dtype=object).value_counts()
            rows.append({
                "drawing": drawing["drawing"],
                "nodes": len(target),
                "written": len(written),
                "right": len(right),
                "construction flag": kinds.get("construction flag", 0),
                "class": kinds.get("class", 0),
                f"off by {TOLERANCE}–{LOOSE} px": kinds.get(f"off by {TOLERANCE}–{LOOSE} px", 0),
                "not written": kinds.get("not written", 0),
                "duplicate": sum(any(other["class"] == node["class"] and isAt(node, other, TOLERANCE) for other in right) for node in extra),
                # Whether each written node is right, in the order it was written.
                "rightInOrder": [at is not None for at in found],
            })
    rows = pd.DataFrame(rows)
    rows["missed"] = rows.nodes - rows.right
    rows["extra"] = rows.written - rows.right
    rows["exact"] = (rows.missed == 0) & (rows.extra == 0)
    return rows


def present(folders):
    return [path for f in folders for path in (RUNS / f).glob(f"*-test-{STEP}.jsonl")]


runs = {key: pd.concat([drawings(path).assign(seed=at) for at, path in enumerate(present(folders))])
        for key, folders in CONFIGURATIONS.items() if present(folders)}
print("\n".join(f"{model:6s} {order:9s} {rows.seed.nunique()} seed(s)" for (model, order), rows in runs.items()))

plt.rcParams.update({
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.color": "#e6e6e3",
    "axes.edgecolor": "#c3c2b7",
    "lines.linewidth": 2,
    "legend.frameon": False,
})


def label(key):
    model, order = key
    return f"{model}, {order}"


# %% Whole sketches right, by how many nodes the sketch holds
fig, axes = plt.subplots(1, 2, figsize=(18, 6))
for key, rows in runs.items():
    bySize = rows.groupby("nodes")
    style = dict(color=HUES[key[0]], linestyle=STYLES[key[1]], label=label(key))
    axes[0].plot(bySize.exact.mean() * 100, **style)
    axes[1].plot(bySize.apply(lambda size: np.maximum(size.missed, size.extra).mean()), **style)
sketches = next(iter(runs.values())).query("seed == 0").nodes.value_counts().sort_index()
for nodes, count in sketches.items():
    axes[0].annotate(f"{count}", (nodes, 1), ha="center", fontsize=6, color="#888780")
axes[0].set_title(f"sketches exactly right at {TOLERANCE} px (grey: how many sketches)", loc="left")
axes[0].set_ylabel("%")
axes[0].set_ylim(0, 100)
axes[1].set_title("nodes to put right per sketch", loc="left")
for ax in axes:
    ax.set_xlabel("nodes in the sketch")
axes[0].legend(fontsize=8)
fig.suptitle("Vitruvion primitives, test, mean over seeds", x=0.01, ha="left")
fig.tight_layout()
plt.show()


# %% The kinds of mistake, as % of target nodes
MISSED = ["construction flag", "class", f"off by {TOLERANCE}–{LOOSE} px", "not written"]
kinds = pd.DataFrame({
    label(key): {
        **{f"missed: {kind}": rows[kind].sum() / rows.nodes.sum() * 100 for kind in MISSED},
        "extra: duplicate": rows.duplicate.sum() / rows.nodes.sum() * 100,
        "extra: other": (rows.extra.sum() - rows.duplicate.sum() - rows[MISSED[:3]].sum().sum()) / rows.nodes.sum() * 100,
        "too few written (% of sketches)": (rows.written < rows.nodes).mean() * 100,
        "too many written (% of sketches)": (rows.written > rows.nodes).mean() * 100,
        "exact (% of sketches)": rows.exact.mean() * 100,
    }
    for key, rows in runs.items()
})
print(f"at {TOLERANCE} px; an extra node in the place of a missed one is counted once, as the missed one's kind\n")
print(kinds.round(2).to_string())

fig, ax = plt.subplots(figsize=(12, 5))
shown = kinds.loc[[f"missed: {kind}" for kind in MISSED] + ["extra: duplicate", "extra: other"]].T
shown.plot.barh(stacked=True, ax=ax, width=0.7, color=["#b9b8ae", "#7d7c74", "#e8b04b", "#c0392b", "#6a8fd8", "#2c3e7a"], edgecolor="white", linewidth=2)
ax.invert_yaxis()
ax.set_xlabel(f"% of target nodes, at {TOLERANCE} px")
ax.legend(fontsize=8, loc="lower right")
fig.tight_layout()
plt.show()


# %% Where in its own writing a model goes wrong: the share of the k-th written node that is wrong
ORDERED = [key for key in runs if key[0] in ("d2s", "d2tok", "d2seq")]
fig, axes = plt.subplots(1, 2, figsize=(18, 6))
for key in ORDERED:
    rows = runs[key]
    style = dict(color=HUES[key[0]], linestyle=STYLES[key[1]], label=label(key))
    positions = pd.DataFrame([(k, isRight) for written in rows.rightInOrder for k, isRight in enumerate(written)], columns=["k", "right"])
    byPosition = positions.groupby("k").right
    shown = byPosition.count() >= 200
    axes[0].plot((100 - byPosition.mean() * 100)[shown], **style)
    # Measured against how far into its own sketch: the first wrong node, of the sketches with one.
    first = rows.rightInOrder.map(lambda written: next((k for k, isRight in enumerate(written) if not isRight), None)).dropna()
    relative = first / rows.written[first.index]
    axes[1].hist(relative, bins=np.linspace(0, 1, 11), histtype="step", linewidth=2, density=True, **style)
axes[0].set_title("written nodes that are wrong, by when they were written", loc="left")
axes[0].set_xlabel("position in the written sequence (nodes)")
axes[0].xaxis.get_major_locator().set_params(integer=True)
axes[0].set_ylabel("%")
axes[1].set_title("the first wrong node, by how far into its writing (sketches with one)", loc="left")
axes[1].set_xlabel("position / nodes written")
axes[1].set_ylabel("density")
axes[0].legend(fontsize=8)
fig.tight_layout()
plt.show()


# %% The same sketches: does d2tok get right what d2s in class order gets right?
pair = [("d2tok", "random"), ("d2s", "class")]
if all(key in runs for key in pair):
    exact = [runs[key].query("seed == 0").set_index("drawing").exact for key in pair]
    both = pd.crosstab(exact[0], exact[1], normalize=True) * 100
    both.index = [f"{label(pair[0])} wrong", f"{label(pair[0])} right"]
    both.columns = [f"{label(pair[1])} wrong", f"{label(pair[1])} right"]
    print(f"\n% of test sketches, seed folder 0 of each\n{both.round(1).to_string()}")


# %% PF1 as DAVINCI counts it (https://arxiv.org/abs/2410.22857, section 4): the first 5,000 test
# sketches, a primitive found when its class and every coordinate are within 5 of the 64 bins a
# sketch unit is quantised to, the construction flag ignored; pooled over the sketches, mean over
# seeds. A sketch unit spans 83.378 / 1.2 px of the render.
DAVINCI_SKETCHES, DAVINCI_TOLERANCE = 5000, 5 / 64 * 83.378 / 1.2


def davinci(path):
    found, targets, written = 0, 0, 0
    with open(path) as lines:
        for line, _ in zip(lines, range(DAVINCI_SKETCHES)):
            drawing = json.loads(line)
            target, nodes = drawing["target"]["nodes"], drawing["predicted"]["nodes"]
            taken = set()
            for node in target:
                slot = next((at for at, other in enumerate(nodes) if at not in taken and other["class"] == node["class"]
                             and isAt(node, other, DAVINCI_TOLERANCE)), None)
                if slot is not None:
                    taken.add(slot)
            found, targets, written = found + len(taken), targets + len(target), written + len(nodes)
    recall, precision = found / targets * 100, found / written * 100
    return {"found": recall, "right": precision, "PF1": 2 * recall * precision / (recall + precision)}


print(pd.DataFrame({label(key): pd.DataFrame([davinci(path) for path in present(folders)]).mean()
                    for key, folders in CONFIGURATIONS.items() if present(folders)}).T.round(1).to_string())
