# %% What to look at
# Run cell by cell in an interactive window, or the whole file for every plot.
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

RESULT = Path("d2g-rectilinear-s-deep.csv")

# A model that decides by a threshold is scored at several; read it at this one. None for a
# model that has no threshold.
THRESHOLD = None

# %% Load
scores = pd.read_csv(RESULT)
scores = scores[scores.threshold == THRESHOLD] if THRESHOLD is not None else scores[scores.threshold.isna()]
scores = scores.sort_values(["step", "tolerance"])
tolerances = sorted(scores.tolerance.unique())
last = scores.step.max()

# The line each score gets, in the words of the score report.
SCORES = {
    "node_recall": "nodes found",
    "node_precision": "nodes right",
    "nodes_exact": "nodes exactly right",
    "edge_recall": "relationships found",
    "edge_precision": "relationships right",
    "records_exact": "records exactly right",
}

run = f"{scores.model.iloc[0]} on {scores.corpus.iloc[0]}, {scores['size'].iloc[0]}"
print(run)
print(f"{scores.parameters.iloc[0]:,} parameters, {scores.training_seconds.iloc[0] / 3600:.1f} h to step {last:,}")
with pd.option_context("display.width", 200):
    print(scores[scores.step == last].set_index("tolerance")[list(SCORES)].rename(columns=SCORES))

# One colour per tolerance, in a fixed order, so a tolerance keeps its colour across plots.
COLOURS = dict(zip(tolerances, ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]))

plt.rcParams.update({
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.color": "#e6e6e3",
    "axes.edgecolor": "#c3c2b7",
    "lines.linewidth": 2,
    "lines.markersize": 6,
    "legend.frameon": False,
})


def step_axis(ax):
    ax.set_xlabel("step")
    ax.xaxis.set_major_formatter(lambda step, _: f"{step / 1000:.0f}k")


def label_ends(ax, ends):
    """The last value of each line, written after it; lines that end close together get their
    labels pushed apart, bottom up, so every one can be read."""
    gap = 4.5
    placed = []
    for value in sorted(ends):
        placed.append(value if not placed else max(value, placed[-1] + gap))
    for value, at in zip(sorted(ends), placed):
        ax.annotate(f"{value:.0f}%", (last, value), xytext=(8, (at - value) * 2.4), textcoords="offset points",
                    va="center", fontsize=9, color="#52514e")


# %% Every score over the checkpoints, one line per tolerance
fig, axes = plt.subplots(2, 3, figsize=(15, 8), sharex=True, sharey=True)
for ax, (score, what) in zip(axes.flat, SCORES.items()):
    for tolerance, at in scores.groupby("tolerance"):
        ax.plot(at.step, at[score], marker="o", color=COLOURS[tolerance], label=f"{tolerance:.0f} px")
    label_ends(ax, scores[scores.step == last][score])
    ax.set_title(what, loc="left")
    ax.set_ylim(0, 100)
    ax.set_xlim(0, last * 1.12)
for ax in axes[-1]:
    step_axis(ax)
for ax in axes[:, 0]:
    ax.set_ylabel("%")
axes[0, 0].legend(title="tolerance", loc="lower right")
fig.suptitle(run, x=0.01, ha="left")
fig.tight_layout()
plt.show()

# %% How much each checkpoint added to whole records right — is it still learning?
gained = scores.pivot(index="step", columns="tolerance", values="records_exact").diff()
fig, ax = plt.subplots(figsize=(9, 4))
for tolerance in tolerances:
    ax.plot(gained.index, gained[tolerance], marker="o", color=COLOURS[tolerance], label=f"{tolerance:.0f} px")
ax.axhline(0, color="#c3c2b7", linewidth=1)
ax.set_title(f"{run} — records exactly right, gained per checkpoint", loc="left")
ax.set_ylabel("percentage points")
step_axis(ax)
ax.legend(title="tolerance")
fig.tight_layout()
plt.show()

# %% Where a threshold decides: the last checkpoint at every threshold
thresholded = pd.read_csv(RESULT).dropna(subset=["threshold"])
if thresholded.empty:
    print("no threshold in this run")
else:
    at_last = thresholded[thresholded.step == thresholded.step.max()].sort_values("threshold")
    fig, ax = plt.subplots(figsize=(9, 5))
    for tolerance, at in at_last.groupby("tolerance"):
        ax.plot(at.threshold, at.records_exact, marker="o", color=COLOURS[tolerance], label=f"{tolerance:.0f} px")
    ax.set_title(f"{run} — records exactly right at step {at_last.step.iloc[0]:,}", loc="left")
    ax.set_xlabel("threshold")
    ax.set_ylabel("%")
    ax.set_ylim(0, 100)
    ax.legend(title="tolerance")
    fig.tight_layout()
    plt.show()
