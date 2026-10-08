# %% What to look at
# DETR with 16 queries against d2s on sketch-xl, at every learning rate tried: batch 512, Adam β₂ 0.98
# and weight decay 0.05 for all of them, as sync_results.sh fetches them into results/runs/.
# Run cell by cell in an interactive window, or the whole file for every plot.
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

RUNS = Path("results/runs")

# A run by its model and learning rate: the folder it was written to.
FOLDERS = {
    ("d2s", "1e-4"): "xl-512-lr1e-4-d2s-b098-wd05",
    ("d2s", "3e-4"): "xl-512-lr3e-4-d2s-b098-wd05",
    ("d2s", "6e-4"): "xl-512-lr6e-4-d2s-b098-wd05",
    ("detr", "1e-4"): "xl-512-lr1e-4-detr-q16-b098-wd05",
    ("detr", "3e-4"): "xl-512-lr3e-4-detr-q16-b098-wd05",
    ("detr", "6e-4"): "xl-512-lr6e-4-detr-q16-b098-wd05",
}

# The tolerance the runs are compared at, in pixels.
TOLERANCE = 4

# A model is a hue, a higher learning rate a darker shade of it.
COLOURS = {
    ("d2s", "1e-4"): "#f0997b", ("d2s", "3e-4"): "#d85a30", ("d2s", "6e-4"): "#712b13",
    ("detr", "1e-4"): "#85b7eb", ("detr", "3e-4"): "#378add", ("detr", "6e-4"): "#0c447c",
}


# %% Load
def scores(folder):
    """Every scored checkpoint at the tolerance compared at, in step order."""
    rows = pd.read_csv(next(folder.glob("*-s-deep.csv")))
    rows = rows[rows.threshold.isna() & (rows.tolerance == TOLERANCE)].sort_values("step")
    rows["node_f1"] = 2 * rows.node_recall * rows.node_precision / (rows.node_recall + rows.node_precision)
    return rows


def history(folder):
    return pd.read_csv(sorted(folder.glob("checkpoints/*/*/*/history.csv"))[-1])


scored = {run: scores(RUNS / folder) for run, folder in FOLDERS.items()}
histories = {run: history(RUNS / folder) for run, folder in FOLDERS.items()}

summary = pd.DataFrame([
    {
        "model": model,
        "learning rate": rate,
        "checkpoints scored": len(s),
        "records exact, last": s.records_exact.iloc[-1],
        "records exact, best": s.records_exact.max(),
        "best at": int(s.step.loc[s.records_exact.idxmax()]),
        "nodes found": s.node_recall.iloc[-1],
        "nodes right": s.node_precision.iloc[-1],
        "node F1": round(s.node_f1.iloc[-1], 1),
    }
    for (model, rate), s in scored.items()
])
with pd.option_context("display.width", 200, "display.max_columns", 20):
    print(f"at {TOLERANCE} px\n{summary.to_string(index=False)}")

plt.rcParams.update({
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.color": "#e6e6e3",
    "axes.edgecolor": "#c3c2b7",
    "lines.linewidth": 2,
    "legend.frameon": False,
})


def step_axis(ax):
    ax.set_xlabel("step")
    ax.xaxis.set_major_formatter(lambda step, _: f"{step / 1000:.0f}k")


def label(run):
    model, rate = run
    return f"{'DETR, 16 queries' if model == 'detr' else 'd2s'}, lr {rate}"


# %% What the checkpoints score: whole sketches exactly right, and nodes
fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharex=True)
for ax, (score, what) in zip(axes, [("records_exact", "sketches exactly right"), ("node_f1", "node F1")]):
    for run, s in scored.items():
        ax.plot(s.step, s[score], color=COLOURS[run], marker="o", markersize=3, label=label(run))
    ax.set_title(f"{what} at {TOLERANCE} px", loc="left")
    ax.set_ylabel("%")
    step_axis(ax)
axes[0].set_ylim(0, 70)
axes[1].set_ylim(50, 100)
axes[0].legend(fontsize=9, loc="lower right")
fig.suptitle("sketch-xl, s-deep, batch 512, β₂ 0.98, weight decay 0.05 — scored on the validation split", x=0.01, ha="left")
fig.tight_layout()
plt.show()

# %% The training loss of every run, which is not comparable between the two models
fig, axes = plt.subplots(1, 2, figsize=(16, 5), sharex=True)
for ax, model in zip(axes, ["d2s", "detr"]):
    for run, h in histories.items():
        if run[0] == model:
            ax.plot(h.step, h.train_loss, color=COLOURS[run], label=label(run))
    ax.set_yscale("log")
    ax.set_title(label((model, "")).rsplit(",", 1)[0], loc="left")
    ax.set_ylabel("training loss (running average)")
    step_axis(ax)
    ax.legend(fontsize=9)
fig.tight_layout()
plt.show()
