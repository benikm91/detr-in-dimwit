# How every model trained on the synthetic floor plans, 200k steps: whole plans right on validation
# at every checkpoint, and the training loss, a line per model and order, averaged over its seeds
# with their range shaded. Writes synthetic-floorplans-training.png.
#
#   uv run --with matplotlib --with pandas python plot_fp.py
from itertools import accumulate
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

RUNS = Path("results/runs")
OUT = Path("synthetic-floorplans-training.png")

SEEDS = (0, 1, 2)
# A model and order, and its runs: `<stem>-seed<seed>-200k`. Seed 0's 200k runs carry on from the
# checkpoint at 80k of `<stem>-seed0`, whose eval scored the checkpoints they share before it.
CONFIGURATIONS = {
    ("DETR", "–"): "floorplans-detr",
    ("d2s", "free"): "floorplans-d2s-free",
    ("d2s", "walls first"): "floorplans-d2s-walls-first",
    ("d2tok", "random"): "floorplans-d2tok-random",
    ("d2tok", "walls first"): "floorplans-d2tok-walls-first",
}
# The tolerance the runs are compared at, in pixels of the 128 canvas the models read.
TOLERANCE = 4

# Where the learning rate starts falling to zero.
COOLDOWN = 160_000

# A model is a hue, as in plot_vitruvion.py; a model without an order of its classes is dashed.
HUES = {"d2s": "#2a78d6", "DETR": "#eb6834", "d2tok": "#1baf7a"}
DASHED = {"free", "random"}


def scored(seed_run):
    shared, run = seed_run
    def read(folder):
        rows = pd.read_csv(next(folder.glob("*-val.csv")))
        return rows[rows.threshold.isna() & (rows.tolerance == TOLERANCE)]
    rows = read(run)
    if shared.exists():
        rows = pd.concat([read(shared).query("step not in @rows.step"), rows])
    return rows.sort_values("step").set_index("step")


def history(seed_run):
    _, run = seed_run
    return pd.read_csv(sorted(run.glob("checkpoints/*/*/*/history.csv"))[-1]).set_index("step")


def present(stem):
    return [(RUNS / f"{stem}-seed{s}", RUNS / f"{stem}-seed{s}-200k") for s in SEEDS
            if any((RUNS / f"{stem}-seed{s}-200k").glob("*-val.csv"))]


runs = {key: present(stem) for key, stem in CONFIGURATIONS.items()}
runs = {key: seeds for key, seeds in runs.items() if seeds}

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
    return model if order == "–" else f"{model}, {order}"


def across_seeds(frames, column):
    """The mean and range of a column over the seeds, at the steps every seed has."""
    joined = pd.concat([frame[column] for frame in frames], axis=1, join="inner")
    return joined.mean(axis=1), joined.min(axis=1), joined.max(axis=1)


def line(ax, key, mean, low, high):
    colour = HUES[key[0]]
    ax.plot(mean.index, mean, color=colour, linestyle="--" if key[1] in DASHED else "-", marker="o", markersize=3, label=label(key))
    ax.fill_between(mean.index, low, high, color=colour, alpha=0.15, linewidth=0)
    ax.axvline(COOLDOWN, color="#c3c2b7", linewidth=1)
    ax.xaxis.set_major_formatter(lambda step, _: f"{step / 1000:.0f}k")


fig = plt.figure(figsize=(14, 9), layout="constrained")
top, bottom = fig.subfigures(2, 1, height_ratios=[1.3, 1])

# Whole plans right, every line labelled at its end, the labels of lines ending close together
# pushed apart.
ax = top.subplots()
ends = []
for key, seeds in runs.items():
    mean, low, high = across_seeds([scored(seed) for seed in seeds], "records_exact")
    line(ax, key, mean, low, high)
    ends.append((mean.index[-1], mean.iloc[-1], label(key)))
ax.set_ylim(0, 100)
ax.set_xlim(right=230_000)
height = (ax.transScale + ax.transLimits).transform([(x, y) for x, y, _ in ends])[:, 1]
ends = sorted(zip(height, ends))
gap = 0.05
upwards = list(accumulate([h for h, _ in ends], lambda below, h: max(h, below + gap)))
placed = reversed(list(accumulate(reversed(upwards), lambda above, h: min(h, above - gap), initial=1 + gap))[1:])
for at, (_, (x, y, text)) in zip(placed, ends):
    ax.annotate(f"{text}  {y:.1f}", (x, y), xytext=(x + 3000, at), textcoords=ax.get_xaxis_transform(),
                va="center", fontsize=9, color="#52514e")
ax.set_title(f"plans exactly right at {TOLERANCE} px, validation (5,000 plans)", loc="left")
ax.set_ylabel("%")
ax.set_xlabel("step (grey line: the cooldown begins)")
ax.legend(fontsize=9, loc="lower right")

# The training loss, a panel per model: the losses of different models are not on one scale.
models = list(dict.fromkeys(model for model, _ in runs))
for ax, model in zip(bottom.subplots(1, len(models), sharex=True), models):
    for key, seeds in runs.items():
        if key[0] == model:
            line(ax, key, *across_seeds([history(seed) for seed in seeds], "train_loss"))
    ax.set_yscale("log")
    ax.set_title(f"{model}: training loss", loc="left")
    ax.set_xlabel("step")
    if sum(key[0] == model for key in runs) > 1:
        ax.legend(fontsize=8)

seeds = "/".join(map(str, sorted({len(s) for s in runs.values()})))
fig.suptitle(f"Synthetic floor plans, s-deep at 128 px, 200k steps, {seeds} seed per line (the range shaded where more)", x=0.01, ha="left")
fig.savefig(OUT, dpi=150)
print(OUT)
