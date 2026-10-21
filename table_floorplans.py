# The tables of synthetic-floorplans-results.md: every configuration's last checkpoint after 200k
# steps, mean ± standard deviation over the seeds present, from the scores the eval wrote
# (`*-<split>.csv`); and the language model's few test plans beside the trained models' transcripts
# of the same plans (`*-test-<step>.jsonl`).
#
#   python3 table_floorplans.py
import csv
import json
import statistics
from pathlib import Path

RUNS = Path("results/runs")

SEEDS = (0, 1, 2)
# A configuration's runs are `<stem>-seed<seed>-200k`.
CONFIGURATIONS = [
    ("DETR", "–", "floorplans-detr"),
    ("d2s", "free", "floorplans-d2s-free"),
    ("d2s", "walls first", "floorplans-d2s-walls-first"),
    ("d2tok", "random", "floorplans-d2tok-random"),
    ("d2tok", "walls first", "floorplans-d2tok-walls-first"),
]

LLM = RUNS / "floorplans-llm-fable-5-1"

# The tolerances are in pixels of the 128 px canvas the models read; the transcripts are in pixels of
# the 512 px drawing.
SCALE = 512 / 128


def last(folder, split):
    """The last checkpoint's rows of a run, by tolerance; nothing if the split is not scored."""
    scores = next(folder.glob(f"*-{split}.csv"), None)
    if scores is None:
        return None
    rows = [row for row in csv.DictReader(open(scores)) if not row["threshold"]]
    step = max(int(row["step"]) for row in rows)
    return {row["tolerance"]: row for row in rows if int(row["step"]) == step}


def spread(values):
    return f"{statistics.mean(values):.1f} ± {statistics.stdev(values):.1f}" if len(values) > 1 else f"{values[0]:.1f}"


def configurations(split):
    """Every configuration with a scored run: its name, and the last rows of its seeds."""
    for model, order, stem in CONFIGURATIONS:
        folders = [RUNS / f"{stem}-seed{s}-200k" for s in SEEDS]
        runs = [rows for rows in (last(folder, split) for folder in folders if folder.exists()) if rows]
        if runs:
            yield model, order, runs


for split in ("test", "val"):
    print(f"\n## {split}, 200k steps, at 4 px\n")
    print("| model | order | parameters | seeds | plans exactly right | node F1 | nodes found | nodes right |")
    print("|---|---|---|---|---|---|---|---|")
    by_tolerance = []
    for model, order, runs in configurations(split):
        found = [float(rows["4"]["node_recall"]) for rows in runs]
        right = [float(rows["4"]["node_precision"]) for rows in runs]
        f1 = [2 * r * p / (r + p) for r, p in zip(found, right)]
        exact = lambda tolerance: [float(rows[tolerance]["records_exact"]) for rows in runs]
        parameters = int(runs[0]["4"]["parameters"]) / 1e6
        print(f"| {model} | {order} | {parameters:.2f}M | {len(runs)} | {spread(exact('4'))} | {spread(f1)} | {spread(found)} | {spread(right)} |")
        by_tolerance.append((model, order, len(runs), [statistics.mean(exact(t)) for t in ("2", "4", "8")]))
    print(f"\nplans exactly right by tolerance ({split})\n")
    print("| model | order | seeds | 2 px | 4 px | 8 px |")
    print("|---|---|---|---|---|---|")
    for model, order, seeds, means in by_tolerance:
        print(f"| {model} | {order} | {seeds} | " + " | ".join(f"{m:.1f}" for m in means) + " |")


def near(node, other, pixels):
    """Of the same class, with both ends within `pixels` in x and in y."""
    return node["class"] == other["class"] and all(abs(a - b) <= pixels for p, q in zip(node["points"], other["points"]) for a, b in zip(p, q))


def matched(target, written, pixels):
    """The written nodes found: every target node, in its order, takes the first one still free near it."""
    taken = set()
    for node in target:
        slot = next((at for at, other in enumerate(written) if at not in taken and near(node, other, pixels)), None)
        if slot is not None:
            taken.add(slot)
    return taken


def on_plans(pairs):
    """Plans exactly right at 2, 4 and 8 px, and nodes found and right at 4 px, in %."""
    def exact(tolerance):
        return sum(len(matched(t, w, tolerance * SCALE)) == len(t) == len(w) for t, w in pairs) / len(pairs) * 100
    found = sum(len(matched(t, w, 4 * SCALE)) for t, w in pairs)
    return [exact(t) for t in (2, 4, 8)], found / sum(len(t) for t, _ in pairs) * 100, found / max(sum(len(w) for _, w in pairs), 1) * 100


transcripts = next(LLM.glob("llm-*-oneshot-test.jsonl"), None)
if transcripts:
    answers = [json.loads(line) for line in open(transcripts)]
    plans = [answer["plan"] for answer in answers]
    table = [("Claude Fable 5.1, one example", [(a["target"]["nodes"], a["predicted"]["nodes"]) for a in answers])]
    for model, order, stem in CONFIGURATIONS:
        folder = RUNS / f"{stem}-seed0-200k"
        written = next(folder.glob("*-test-*.jsonl"), None) if folder.exists() else None
        if written:
            by_plan = {plan["plan"]: plan for plan in map(json.loads, open(written))}
            table.append((f"{model}, {order}".replace(", –", ""), [(by_plan[at]["target"]["nodes"], by_plan[at]["predicted"]["nodes"]) for at in plans]))
    print(f"\n## the language model's {len(plans)} test plans, at 4 px\n")
    print("| model | plans exactly right (2 / 4 / 8 px) | nodes found | nodes right |")
    print("|---|---|---|---|")
    for name, pairs in table:
        exact, found, right = on_plans(pairs)
        print(f"| {name} | " + " / ".join(f"{e:.0f}" for e in exact) + f" | {found:.1f} | {right:.1f} |")


def errors(folder):
    """At 4 px, from the test transcripts: target nodes missed, extra nodes that repeat a found one,
    other extra nodes and extra doors, as % of target nodes; plans with too few and too many nodes, as
    % of plans."""
    missed = repeats = other = doors = targets = few = many = plans = 0
    for plan in map(json.loads, open(next(folder.glob("*-test-*.jsonl")))):
        target, written = plan["target"]["nodes"], plan["predicted"]["nodes"]
        taken = matched(target, written, 4 * SCALE)
        extra = [w for at, w in enumerate(written) if at not in taken]
        repeating = sum(any(near(w, written[at], 4 * SCALE) for at in taken) for w in extra)
        missed, repeats, other = missed + len(target) - len(taken), repeats + repeating, other + len(extra) - repeating
        doors += sum(w["class"] == "door" for w in extra)
        targets, plans = targets + len(target), plans + 1
        few, many = few + (len(written) < len(target)), many + (len(written) > len(target))
    return [missed / targets * 100, repeats / targets * 100, other / targets * 100, doors / targets * 100, few / plans * 100, many / plans * 100]


ERRORS = ["missed", "extra: a repeat of a found node", "extra: other", "extra: of them doors",
          "too few written (% of plans)", "too many written (% of plans)"]
columns = [(f"{model} {order}".replace(" –", ""), errors(RUNS / f"{stem}-seed0-200k")) for model, order, stem in CONFIGURATIONS
           if next((RUNS / f"{stem}-seed0-200k").glob("*-test-*.jsonl"), None)]
print("\n## what the errors are, test, 200k, at 4 px, % of target nodes unless said otherwise\n")
print("| | " + " | ".join(name for name, _ in columns) + " |")
print("|---|" + "---|" * len(columns))
for at, error in enumerate(ERRORS):
    print(f"| {error} | " + " | ".join(f"{values[at]:.2f}" for _, values in columns) + " |")
