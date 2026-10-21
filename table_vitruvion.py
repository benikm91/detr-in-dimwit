# The tables of vitruvion-results.md: every configuration's last checkpoint, mean ± standard deviation
# over the seeds present, from the scores the eval wrote (`*-<split>.csv`) and, for PF1, from its
# transcripts (`*-<split>-270000.jsonl`). Prints the test split, then validation for comparison with
# runs whose test is not in yet.
#
#   python3 table_vitruvion.py
import csv
import json
import statistics
from pathlib import Path

RUNS = Path("results/runs")
STEP = "270000"

SEEDS = (0, 1, 42)
CONFIGURATIONS = [
    ("d2seq", "random", [f"vitruvion-d2seq-random-seed{s}" for s in SEEDS]),
    ("d2seq", "kept", [f"vitruvion-d2seq-kept-seed{s}" for s in SEEDS]),
    ("d2s", "free", ["vitruvion-final-d2s", "vitruvion-final-d2s-seed0", "vitruvion-final-d2s-seed1"]),
    ("d2s", "class", ["vitruvion-final-d2po", "vitruvion-final-d2po-seed0", "vitruvion-final-d2po-seed1"]),
    ("d2s", "class reversed", [f"vitruvion-final-d2po-reversed-seed{s}" for s in SEEDS]),
    ("DETR", "–", ["vitruvion-final-detr-q16", "vitruvion-final-detr-q16-seed1", "vitruvion-final-detr-q16-seed42"]),
    ("DETR, buckets", "–", ["vitruvion-detr-bucket-seed0"]),
    ("d2tok", "random", [f"vitruvion-d2tok-random-seed{s}" for s in SEEDS]),
    ("d2tok", "kept", [f"vitruvion-d2tok-kept-seed{s}" for s in SEEDS]),
    ("d2tok", "class", [f"vitruvion-d2tok-class-seed{s}" for s in SEEDS]),
    ("d2tok", "position", [f"vitruvion-d2tok-position-seed{s}" for s in SEEDS]),
]

# PF1 as DAVINCI (https://arxiv.org/abs/2410.22857, section 4) counts it: the first 5,000 sketches, a
# primitive found when its class and every coordinate lie within 5 of the 64 bins a sketch unit is
# quantised to, the construction flag ignored. A sketch unit spans 83.378 / 1.2 px of the render.
DAVINCI_SKETCHES, DAVINCI_TOLERANCE = 5000, 5 / 64 * 83.378 / 1.2


def near(node, other, pixels):
    return all(abs(a - b) <= pixels for at, by in zip(node["points"], other["points"]) for a, b in zip(at, by))


def pf1(transcripts):
    found, targets, written = 0, 0, 0
    with open(transcripts) as lines:
        for line, _ in zip(lines, range(DAVINCI_SKETCHES)):
            sketch = json.loads(line)
            target, nodes = sketch["target"]["nodes"], sketch["predicted"]["nodes"]
            taken = set()
            for node in target:
                slot = next((at for at, other in enumerate(nodes)
                             if at not in taken and other["class"] == node["class"] and near(node, other, DAVINCI_TOLERANCE)), None)
                if slot is not None:
                    taken.add(slot)
            found, targets, written = found + len(taken), targets + len(target), written + len(nodes)
    recall, precision = found / targets * 100, found / written * 100
    return 2 * recall * precision / (recall + precision)


def scored(folder, split):
    """The last checkpoint's rows of a run, by tolerance, and its PF1; nothing if the split is not scored."""
    scores = next(folder.glob(f"*-{split}.csv"), None)
    if scores is None:
        return None
    rows = {row["tolerance"]: row for row in csv.DictReader(open(scores)) if row["step"] == STEP and not row["threshold"]}
    if "4" not in rows:
        return None
    transcripts = next(folder.glob(f"*-{split}-{STEP}.jsonl"), None)
    return rows, pf1(transcripts) if transcripts else None


def spread(values):
    return f"{statistics.mean(values):.1f} ± {statistics.stdev(values):.1f}" if len(values) > 1 else f"{values[0]:.1f}"


for split in ("test", "val"):
    print(f"\n## {split}, step {STEP}, at 4 px\n")
    print("| model | order | parameters | seeds | sketches exactly right | node F1 | nodes found | nodes right | PF1 |")
    print("|---|---|---|---|---|---|---|---|---|")
    by_tolerance = []
    for model, order, folders in CONFIGURATIONS:
        runs = [run for run in (scored(RUNS / folder, split) for folder in folders if (RUNS / folder).exists()) if run]
        if not runs:
            continue
        found = [float(rows["4"]["node_recall"]) for rows, _ in runs]
        right = [float(rows["4"]["node_precision"]) for rows, _ in runs]
        f1 = [2 * r * p / (r + p) for r, p in zip(found, right)]
        exact = lambda tolerance: [float(rows[tolerance]["records_exact"]) for rows, _ in runs]
        pf1s = [value for _, value in runs if value is not None]
        parameters = int(runs[0][0]["4"]["parameters"]) / 1e6
        print(f"| {model} | {order} | {parameters:.2f}M | {len(runs)} | {spread(exact('4'))} | {spread(f1)} | {spread(found)} | {spread(right)} | "
              f"{statistics.mean(pf1s):.1f} |" if pf1s else "– |")
        by_tolerance.append((model, order, len(runs), [statistics.mean(exact(t)) for t in ("2", "4", "8")]))
    print(f"\nsketches exactly right by tolerance ({split})\n")
    print("| model | order | seeds | 2 px | 4 px | 8 px |")
    print("|---|---|---|---|---|---|")
    for model, order, seeds, means in by_tolerance:
        print(f"| {model} | {order} | {seeds} | " + " | ".join(f"{m:.1f}" for m in means) + " |")
