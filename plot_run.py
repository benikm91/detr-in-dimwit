# %% What to look at
# Run cell by cell in an interactive window, or the whole file for every plot.
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

# The runs to look at; the order fixes which colour each of them gets where they are compared.
RUNS = [Path("d2g-rectilinear-s-deep.csv"), Path("detr-rectilinear-s-deep.csv")]

# A model that decides by a threshold is scored at several; read it at this one. None for a
# model that has no threshold.
THRESHOLD = None

# The line each score gets, in the words of the score report.
SCORES = {
    "node_recall": "nodes found",
    "node_precision": "nodes right",
    "nodes_exact": "nodes exactly right",
    "edge_recall": "relationships found",
    "edge_precision": "relationships right",
    "records_exact": "records exactly right",
}


# %% Load
def scored(path):
    """The rows of one run at the threshold asked for, in the order the plots read them."""
    scores = pd.read_csv(path)
    scores = scores[scores.threshold == THRESHOLD] if THRESHOLD is not None else scores[scores.threshold.isna()]
    return scores.sort_values(["step", "tolerance"])


def named(scores):
    return f"{scores.model.iloc[0]} on {scores.corpus.iloc[0]}, {scores['size'].iloc[0]}"


runs = {path: scored(path) for path in RUNS}
tolerances = sorted({tolerance for scores in runs.values() for tolerance in scores.tolerance.unique()})

for scores in runs.values():
    last = scores.step.max()
    # A run cut short by its time limit never gets to note how long it trained.
    seconds = scores.training_seconds.iloc[0]
    trained = f"{seconds / 3600:.1f} h" if pd.notna(seconds) else "an unrecorded time"
    print(named(scores))
    print(f"{scores.parameters.iloc[0]:,} parameters, {trained} to step {last:,}")
    with pd.option_context("display.width", 200):
        print(scores[scores.step == last].set_index("tolerance")[list(SCORES)].rename(columns=SCORES))
    print()

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


def label_ends(ax, ends, last):
    """The last value of each line, written after it; lines that end close together get their
    labels pushed apart, bottom up, so every one can be read. A score a model never answers for
    has no end to label."""
    gap = 4.5
    ends = sorted(value for value in ends if pd.notna(value))
    placed = []
    for value in ends:
        placed.append(value if not placed else max(value, placed[-1] + gap))
    for value, at in zip(ends, placed):
        ax.annotate(f"{value:.0f}%", (last, value), xytext=(8, (at - value) * 2.4), textcoords="offset points",
                    va="center", fontsize=9, color="#52514e")


# %% Every score over the checkpoints, one line per tolerance
for scores in runs.values():
    last = scores.step.max()
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), sharex=True, sharey=True)
    for ax, (score, what) in zip(axes.flat, SCORES.items()):
        for tolerance, at in scores.groupby("tolerance"):
            ax.plot(at.step, at[score], marker="o", color=COLOURS[tolerance], label=f"{tolerance:.0f} px")
        label_ends(ax, scores[scores.step == last][score], last)
        if not scores[score].notna().any():
            ax.annotate("not answered", (0.5, 0.5), xycoords="axes fraction", ha="center",
                        va="center", fontsize=9, color="#8a8a85")
        ax.set_title(what, loc="left")
        ax.set_ylim(0, 100)
        ax.set_xlim(0, last * 1.12)
    for ax in axes[-1]:
        step_axis(ax)
    for ax in axes[:, 0]:
        ax.set_ylabel("%")
    axes[0, 0].legend(title="tolerance", loc="lower right")
    fig.suptitle(named(scores), x=0.01, ha="left")
    fig.tight_layout()
    plt.show()

# %% How much each checkpoint added to whole records right — is it still learning?
for scores in runs.values():
    gained = scores.pivot(index="step", columns="tolerance", values="records_exact").diff()
    fig, ax = plt.subplots(figsize=(9, 4))
    for tolerance in gained.columns:
        ax.plot(gained.index, gained[tolerance], marker="o", color=COLOURS[tolerance], label=f"{tolerance:.0f} px")
    ax.axhline(0, color="#c3c2b7", linewidth=1)
    ax.set_title(f"{named(scores)} — records exactly right, gained per checkpoint", loc="left")
    ax.set_ylabel("percentage points")
    step_axis(ax)
    ax.legend(title="tolerance")
    fig.tight_layout()
    plt.show()

# %% Where a threshold decides: the last checkpoint at every threshold
for path, scores in runs.items():
    thresholded = pd.read_csv(path).dropna(subset=["threshold"])
    if thresholded.empty:
        print(f"no threshold in {named(scores)}")
        continue
    at_last = thresholded[thresholded.step == thresholded.step.max()].sort_values("threshold")
    fig, ax = plt.subplots(figsize=(9, 5))
    for tolerance, at in at_last.groupby("tolerance"):
        ax.plot(at.threshold, at.records_exact, marker="o", color=COLOURS[tolerance], label=f"{tolerance:.0f} px")
    ax.set_title(f"{named(scores)} — records exactly right at step {at_last.step.iloc[0]:,}", loc="left")
    ax.set_xlabel("threshold")
    ax.set_ylabel("%")
    ax.set_ylim(0, 100)
    ax.legend(title="tolerance")
    fig.tight_layout()
    plt.show()

# %% The runs on the same corpus, what they say about nodes
compared = pd.concat(runs.values())
# At no tolerance at all nothing is ever exactly right, so that panel would be empty.
compared = compared[compared.tolerance > 0].sort_values(["model", "tolerance", "step"])
# Runs need not be the same length, so each line is named with the step it reaches.
reached = compared.groupby("model").step.max()
models = list(dict.fromkeys(compared.model))
MODEL_COLOURS = dict(zip(models, ["#2a78d6", "#eb6834"]))
# Both models can finish on the same value, so their end labels are nudged apart.
LABEL_NUDGES = dict(zip(models, [7, -7]))

# A detector answers with no relationships, so only what a model says about nodes is comparable.
COMPARED_SCORES = ["node_recall", "node_precision", "nodes_exact"]
tolerances_compared = sorted(compared.tolerance.unique())

fig, axes = plt.subplots(len(COMPARED_SCORES), len(tolerances_compared), sharex=True, sharey=True,
                         figsize=(5 * len(tolerances_compared), 3.6 * len(COMPARED_SCORES)))
for row, score in zip(axes, COMPARED_SCORES):
    for ax, tolerance in zip(row, tolerances_compared):
        for model, at in compared[compared.tolerance == tolerance].groupby("model"):
            ax.plot(at.step, at[score], color=MODEL_COLOURS[model],
                    label=f"{model} (to {reached[model] / 1000:.0f}k)")
            end = at.iloc[-1]
            ax.annotate(f"{end[score]:.0f}%", (end.step, end[score]), xytext=(8, LABEL_NUDGES[model]),
                        textcoords="offset points", va="center", fontsize=9, color="#52514e")
        ax.set_ylim(0, 100)
        ax.set_xlim(0, compared.step.max() * 1.12)
    row[0].set_ylabel(f"{SCORES[score]} (%)")
for ax, tolerance in zip(axes[0], tolerances_compared):
    ax.set_title(f"{tolerance:.0f} px", loc="left")
for ax in axes[-1]:
    step_axis(ax)
axes[0, 0].legend(loc="lower right")
fig.suptitle(f"{compared.corpus.iloc[0]}, {compared['size'].iloc[0]}", x=0.01, ha="left")
fig.tight_layout()
plt.show()
