# %% What to look at
# The sketch-xl runs at a batch of 512, as sync_results.sh fetches them into results/runs/, and the
# batch-128 runs they replaced.
# Run cell by cell in an interactive window, or the whole file for every plot.
import re
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

RUNS = Path("results/runs")

# Each run is a folder named for what sets it apart: xl-512-lr<rate>-<model>[-<variant>][-b098][-wd05].
FOLDERS = sorted(RUNS.glob("xl-512-*"))

# The runs at a batch of 128 that came before: learning rate 3e-4, Adam β₂ 0.999, weight decay 1e-4,
# fp32, and as many drawings seen as the runs at 512.
BATCH_128 = {Path("d2g-sketch-xl-s-deep.csv"): "d2g", Path("detr-sketch-xl-s-deep.csv"): "detr 48 queries"}

# The tolerance the runs are compared at, in pixels.
TOLERANCE = 4


# %% Load
def described(folder):
    """What a run's folder name says about it: its model, learning rate, variant and optimizer."""
    rate, model, rest = re.fullmatch(r"xl-512-lr(\d+e-\d+)-([a-z0-9]+)-?(.*)", folder.name).groups()
    tags = rest.split("-") if rest else []
    optimizer = f"β₂ {'0.98' if 'b098' in tags else '0.999'}, wd {'0.05' if 'wd05' in tags else '1e-4'}"
    variant = " ".join(tag for tag in tags if tag not in ("b098", "wd05", "fp16"))
    return model, rate, variant, optimizer


def named(folder):
    model, rate, variant, optimizer = described(folder)
    return f"{model} {rate}{' ' + variant if variant else ''}, {optimizer}"


def history(folder):
    """The training loss at every checkpoint, or None for a run that saved none."""
    found = sorted(folder.glob("checkpoints/*/*/*/history.csv"))
    return pd.read_csv(found[-1]) if found else None


def scores(folder):
    """The scores at every checkpoint at the tolerance compared at, or None before the eval has run."""
    found = [path for path in folder.glob("*.csv")]
    return scored_in(found[0]) if found else None


def scored_in(path):
    rows = pd.read_csv(path)
    rows = rows[rows.threshold.isna() & (rows.tolerance == TOLERANCE)].sort_values("step")
    if rows.empty:
        return None
    rows["node_f1"] = 2 * rows.node_recall * rows.node_precision / (rows.node_recall + rows.node_precision)
    return rows


histories = {folder: history(folder) for folder in FOLDERS}
histories = {folder: h for folder, h in histories.items() if h is not None}
scored = {folder: scores(folder) for folder in FOLDERS}
scored = {folder: s for folder, s in scored.items() if s is not None}
batch_128 = {name: scored_in(path) for path, name in BATCH_128.items()}

summary = pd.DataFrame([
    {
        "run": named(folder),
        "last step": int(s.step.iloc[-1]),
        "records exact, last": s.records_exact.iloc[-1],
        "records exact, best": s.records_exact.max(),
        "best at": int(s.step.loc[s.records_exact.idxmax()]),
        "nodes found": s.node_recall.iloc[-1],
        "nodes right": s.node_precision.iloc[-1],
        "node F1": round(s.node_f1.iloc[-1], 1),
    }
    for folder, s in scored.items()
]).sort_values("records exact, last", ascending=False)
with pd.option_context("display.width", 200, "display.max_columns", 20):
    print(f"at {TOLERANCE} px\n{summary.to_string(index=False)}")
print(f"\nnot scored yet: {', '.join(named(folder) for folder in FOLDERS if folder not in scored and folder in histories)}")

plt.rcParams.update({
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.color": "#e6e6e3",
    "axes.edgecolor": "#c3c2b7",
    "lines.linewidth": 2,
    "legend.frameon": False,
})

# A run keeps its colour in every plot; its optimizer is its line style.
COLOURS = dict(zip(FOLDERS, plt.get_cmap("tab20").colors * 2))
STYLES = {"β₂ 0.999, wd 1e-4": ":", "β₂ 0.98, wd 1e-4": "--", "β₂ 0.98, wd 0.05": "-"}


def drawn(ax, folder, steps, values):
    ax.plot(steps, values, color=COLOURS[folder], linestyle=STYLES[described(folder)[3]], label=named(folder))


def step_axis(ax):
    ax.set_xlabel("step")
    ax.xaxis.set_major_formatter(lambda step, _: f"{step / 1000:.0f}k")


# %% The training loss of every run: where one slides away
fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharex=True)
for ax, model in zip(axes, ["d2s", "detr"]):
    for folder, h in histories.items():
        if described(folder)[0] == model:
            drawn(ax, folder, h.step, h.train_loss)
    ax.set_yscale("log")
    ax.set_title(model, loc="left")
    ax.set_ylabel("training loss (running average)")
    step_axis(ax)
    ax.legend(fontsize=8)
fig.suptitle("sketch-xl, s-deep, batch 512 — training loss at every checkpoint", x=0.01, ha="left")
fig.tight_layout()
plt.show()

# %% What the checkpoints score: nodes, and whole sketches exactly right
fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharex=True, sharey=True)
for ax, (score, what) in zip(axes, [("node_f1", "node F1"), ("records_exact", "records exactly right")]):
    for folder, s in scored.items():
        drawn(ax, folder, s.step, s[score])
    ax.set_title(f"{what} at {TOLERANCE} px", loc="left")
    ax.set_ylim(0, 100)
    ax.set_ylabel("%")
    step_axis(ax)
axes[0].legend(fontsize=8, loc="lower right")
fig.suptitle("sketch-xl, s-deep, batch 512 — scored on the validation split", x=0.01, ha="left")
fig.tight_layout()
plt.show()

# %% Batch 128 against 512: the same drawings seen in a quarter of the steps
fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharex=True, sharey=True)
for ax, (score, what) in zip(axes, [("node_f1", "node F1"), ("records_exact", "records exactly right")]):
    for (name, s), colour in zip(batch_128.items(), ["#888780", "#2c2c2a"]):
        ax.plot(s.step * 128, s[score], color=colour, linestyle=":", label=f"{name}, batch 128, β₂ 0.999, wd 1e-4")
    for folder, s in scored.items():
        if described(folder)[3] == "β₂ 0.98, wd 0.05":
            drawn(ax, folder, s.step * 512, s[score])
    ax.set_title(f"{what} at {TOLERANCE} px", loc="left")
    ax.set_ylim(0, 100)
    ax.set_ylabel("%")
    ax.set_xlabel("drawings seen")
    ax.xaxis.set_major_formatter(lambda seen, _: f"{seen / 1e6:.0f}M")
axes[0].legend(fontsize=8, loc="lower right")
fig.suptitle("sketch-xl, s-deep — batch 128 against 512", x=0.01, ha="left")
fig.tight_layout()
plt.show()
