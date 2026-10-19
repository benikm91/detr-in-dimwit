# %% What to look at
# How every model trained on the synthetic floor plans: the validation scores at every checkpoint and
# the training loss, a line per model, averaged over its seeds with their range shaded.
# Run cell by cell in an interactive window, or the whole file for every plot.
from itertools import accumulate
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

RUNS = Path("results/runs")

# A model and order, and the folders of its seeds.
CONFIGURATIONS = {
    ("d2s", "free"): ["floorplans-d2s-free-seed0"],
    ("DETR", "free"): ["floorplans-detr-seed0"],
    ("d2tok", "random"): ["floorplans-d2tok-random-seed0"],
}

# The tolerance the runs are compared at, in pixels of the 128 canvas the models read.
TOLERANCE = 4

# Where the learning rate starts falling to zero.
COOLDOWN = 80_000

# A model is a hue, as in plot_vitruvion.py.
HUES = {"d2s": "#2a78d6", "DETR": "#eb6834", "d2tok": "#1baf7a"}


# %% Load: the seeds that exist of every configuration
def scored(folder):
    rows = pd.read_csv(next(folder.glob("*-val.csv")))
    rows = rows[rows.threshold.isna() & (rows.tolerance == TOLERANCE)].sort_values("step")
    rows["node_f1"] = 2 * rows.node_recall * rows.node_precision / (rows.node_recall + rows.node_precision)
    rows["missed"] = 100 - rows.node_recall
    rows["spurious"] = 100 - rows.node_precision
    return rows.set_index("step")


def history(folder):
    return pd.read_csv(sorted(folder.glob("checkpoints/*/*/*/history.csv"))[-1]).set_index("step")


def present(folders):
    return [RUNS / f for f in folders if (RUNS / f).exists() and any((RUNS / f).glob("*-val.csv"))]


runs = {key: present(folders) for key, folders in CONFIGURATIONS.items()}
runs = {key: folders for key, folders in runs.items() if folders}
print("\n".join(f"{model:6s} {order:8s} {len(folders)} seed(s)" for (model, order), folders in runs.items()))

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


def across_seeds(frames, column):
    """The mean and range of a column over the seeds, at the steps every seed has."""
    joined = pd.concat([frame[column] for frame in frames], axis=1, join="inner")
    return joined.mean(axis=1), joined.min(axis=1), joined.max(axis=1)


def draw(ax, column, read):
    ends = []
    for key, folders in runs.items():
        mean, low, high = across_seeds([read(folder) for folder in folders], column)
        colour = HUES[key[0]]
        ax.plot(mean.index, mean, color=colour, marker="o", markersize=4, label=label(key))
        ax.fill_between(mean.index, low, high, color=colour, alpha=0.15, linewidth=0)
        ends.append((mean.index[-1], mean.iloc[-1], label(key)))
    ax.axvline(COOLDOWN, color="#c3c2b7", linewidth=1)
    ax.xaxis.set_major_formatter(lambda step, _: f"{step / 1000:.0f}k")
    ax.set_xlabel("step (grey line: the cooldown begins)")

    # Every line labelled at its end, the labels of lines ending close together pushed apart.
    ax.autoscale_view()
    height = (ax.transScale + ax.transLimits).transform([(x, y) for x, y, _ in ends])[:, 1]
    ends = sorted(zip(height, ends))
    gap = 0.04
    upwards = list(accumulate([h for h, _ in ends], lambda below, h: max(h, below + gap)))
    placed = reversed(list(accumulate(reversed(upwards), lambda above, h: min(h, above - gap), initial=1 + gap))[1:])
    for at, (_, (x, y, text)) in zip(placed, ends):
        ax.annotate(text, (x, y), xytext=(x + 2000, at), textcoords=ax.get_xaxis_transform(),
                    va="center", fontsize=8, color="#52514e")


# %% Convergence: whole plans right, and node F1, at every checkpoint
fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharex=True)
for ax, (column, what) in zip(axes, [("records_exact", "plans exactly right"), ("node_f1", "node F1")]):
    draw(ax, column, scored)
    ax.set_title(f"{what} at {TOLERANCE} px, validation", loc="left")
    ax.set_ylabel("%")
axes[0].legend(fontsize=8, loc="upper left")
fig.suptitle("Synthetic floor plans, s-deep at 128 px: mean over seeds, range shaded", x=0.01, ha="left")
fig.tight_layout()
plt.show()

# %% Errors at every checkpoint: target nodes missed, and written nodes that are none of the target's
fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharex=True)
for ax, (column, what) in zip(axes, [("missed", "target nodes missed"), ("spurious", "written nodes not in the target")]):
    ax.set_yscale("log")
    draw(ax, column, scored)
    ax.set_title(f"{what} at {TOLERANCE} px, validation", loc="left")
    ax.set_ylabel("% of nodes")
axes[0].legend(fontsize=8, loc="upper right")
fig.tight_layout()
plt.show()

# %% The training loss, a panel per model: the losses of different models are not on one scale
models = list(dict.fromkeys(model for model, _ in runs))
fig, axes = plt.subplots(1, len(models), figsize=(5 * len(models), 4.5), sharex=True)
for ax, model in zip(axes, models):
    for key, folders in runs.items():
        if key[0] != model:
            continue
        mean, low, high = across_seeds([history(folder) for folder in folders], "train_loss")
        ax.plot(mean.index, mean, color=HUES[model], marker="o", markersize=4, label=key[1])
        ax.fill_between(mean.index, low, high, color=HUES[model], alpha=0.15, linewidth=0)
    ax.axvline(COOLDOWN, color="#c3c2b7", linewidth=1)
    ax.set_yscale("log")
    ax.set_title(model, loc="left")
    ax.xaxis.set_major_formatter(lambda step, _: f"{step / 1000:.0f}k")
    ax.legend(fontsize=8)
axes[0].set_ylabel("training loss (running average)")
fig.tight_layout()
plt.show()
