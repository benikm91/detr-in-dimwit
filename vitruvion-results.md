# Vitruvion primitives: results

Transcribing a rendered CAD sketch into its primitives (points, lines, circles and arcs, each with a
construction flag), on the Vitruvion split of SketchGraphs: 6–16 primitives per sketch, deduplicated
on a 64 grid, 128 px renders. Test split: 82,181 sketches.

All models are `s-deep`, about 11.3M parameters, the same document encoder, trained for 270k steps
(learning rate cooldown from 216k), three seeds each. The scores are of the last checkpoint on the
test split, mean ± standard deviation over the seeds.

## The models

| model | what it writes | in which order |
|---|---|---|
| **DETR** | all nodes at once, from 16 queries, Hungarian matching | none |
| **d2s** | the next node, trained against the set of the nodes still to write | none (*free*), or class by class (*class*, *class reversed*) |
| **d2seq** | the next node, one embedding per node, teacher-forced | a fresh random order (*random*), or the corpus's creation order (*kept*) |
| **d2tok** | the next word of the record: a type word, then the coordinate words | *random*, *kept*, as for d2seq; *class*; *position* (top to bottom, left to right) |

Everything is decoded greedily; d2tok's decoding is not constrained to well-formed records.

## Main table

At 4 px a node is found when its class and construction flag are right and every point is within
4 px in x and y; a sketch is exactly right when every target node is found and nothing else is
written. PF1 is counted as [DAVINCI](https://arxiv.org/abs/2410.22857) counts it: the first 5,000
test sketches, a tolerance of 5 of Vitruvion's 64 bins (5.43 px on our renders), the construction
flag ignored.

| model | order | parameters | sketches exactly right | node F1 | nodes found | nodes right | PF1 |
|---|---|---|---|---|---|---|---|
| d2seq | random | 11.41M | 0.1 ± 0.0 | 18.6 ± 0.4 | 22.0 ± 0.6 | 16.2 ± 0.3 | – |
| d2seq | kept | 11.41M | 23.8 ± 0.3 | 70.4 ± 0.4 | 71.2 ± 0.3 | 69.7 ± 0.4 | 73.7 |
| d2s | free | 11.41M | 52.9 ± 0.3 | 86.3 ± 0.2 | 83.2 ± 0.2 | 89.7 ± 0.2 | 88.7 |
| d2s | class | 11.41M | 57.7 ± 0.1 | 88.4 ± 0.1 | 86.0 ± 0.0 | 91.0 ± 0.1 | 90.7 |
| d2s | class reversed | 11.41M | 58.1 ± 0.1 | 88.9 ± 0.1 | 86.5 ± 0.1 | 91.4 ± 0.1 | 91.1 |
| DETR | – | 11.32M | 54.1 ± 0.3 | 89.7 ± 0.1 | 88.8 ± 0.1 | 90.5 ± 0.1 | 91.7 |
| d2tok | random | 11.21M | **61.7 ± 0.0** | **90.5 ± 0.1** | **89.4 ± 0.0** | **91.7 ± 0.1** | **92.4** |
| d2tok | kept | 11.21M | 53.7 ± 0.2 | 86.6 ± 0.1 | 86.4 ± 0.1 | 86.9 ± 0.1 | 88.9 |
| d2tok | class | 11.21M | *running* | | | | |
| d2tok | position | 11.21M | *running* | | | | |
| *DAVINCI, its paper* | | | | | | | *91.7* |

### Sketches exactly right, by tolerance

| model | order | 2 px | 4 px | 8 px |
|---|---|---|---|---|
| d2seq | kept | 22.1 | 23.8 | 25.7 |
| d2s | free | 50.3 | 52.9 | 54.4 |
| d2s | class | 55.3 | 57.7 | 59.0 |
| d2s | class reversed | 55.5 | 58.1 | 59.5 |
| DETR | – | 51.8 | 54.1 | 55.4 |
| d2tok | random | **60.0** | **61.7** | **62.4** |
| d2tok | kept | 51.9 | 53.7 | 54.7 |

## What the errors are

From `analyse_vitruvion_errors.py`, at 4 px, as % of target nodes, mean over seeds. A missed node is
explained by an extra node written in its place where there is one; that extra node is not counted
again.

| | d2seq kept | d2s free | d2s class | DETR | d2tok random | d2tok kept |
|---|---|---|---|---|---|---|
| missed: only the construction flag wrong | 1.7 | 1.3 | 1.3 | 1.3 | 1.3 | 1.6 |
| missed: wrong class in the right place | 0.5 | 0.3 | 0.4 | 0.2 | 0.2 | 0.4 |
| missed: off by 4–8 px | 5.5 | 2.8 | 2.4 | 2.3 | 1.9 | 2.6 |
| missed: not written | 21.0 | 12.4 | 9.9 | 7.5 | 7.2 | 9.1 |
| extra: a duplicate of a found node | 3.7 | 1.4 | 1.2 | 2.1 | 1.6 | 2.3 |
| extra: other | 19.5 | 3.7 | 3.2 | 3.5 | 3.0 | 6.2 |
| too few written (% of sketches) | 22.0 | 30.6 | 25.3 | 19.6 | 18.4 | 18.0 |
| too many written (% of sketches) | 31.7 | 3.4 | 4.0 | 12.1 | 7.3 | 14.3 |

- **The models make the same kinds of mistake.** Most of every model's errors are nodes it doesn't
  write at all. d2tok in random order leads mostly because it leaves out fewer nodes (7.2 against
  9.9 for d2s class). The construction flag costs every model about 1.3 points; that looks like a
  floor set by the data, not by the model.
- **d2s stops too early, and d2tok writes on too long.** d2s writes too few nodes in a quarter of
  the sketches, but rarely too many. d2tok's mistakes are more evenly split, with more duplicates.
  The creation order (*kept*) doubles d2tok's extra nodes.
- **They mostly fail on the same sketches.** Seed folder 0 of each: both are right on 53.7% of the
  sketches and both wrong on 34.1%. Only d2tok random is right on 8.0%, only d2s class on 4.2%.
- **Larger sketches are harder for every model, and the ranking holds at every size from 6 to 16
  nodes.** There's no sign that a random order confuses d2tok more as the sketch grows. Within this
  range, d2tok's lead over d2s class grows from about 1 point at 6 nodes to about 5 at 13–15 nodes.
- **Within a model's own writing, later nodes are wrong more often**, for d2s and d2tok alike. The
  first mistake is spread over the whole sequence, not concentrated at its start.

## Convergence

From `plot_vitruvion.py` (validation, every 10k steps):

- d2tok, in either order, leads from the first checkpoint on. In random order it never loses that
  lead. The creation order learns fastest until about 30k steps, then levels off.
- Every model gains 3–10 points of exact sketches in the cooldown. None of them has converged
  before it.
- d2seq was evaluated only from 140k (random) and 200k (kept) on.

## Is the comparison fair?

- **Parameters:** the same budget, 11.2–11.4M; d2tok is the smallest.
- **Training compute:** the same number of steps, batch size and schedule. Wall clock from the runs
  that were not resumed: d2tok about 47k s, d2seq about 47k s, DETR about 48k s, d2s about 53k s.
  These are on mixed GPUs and only a rough guide.
- **Inference compute:** d2tok runs one decoder step per word, up to 7 words per node (a type word
  and up to 6 coordinate words for an arc). A 16-node sketch can take up to 113 sequential steps.
  d2s takes one step per node, about 17. DETR takes one parallel pass. With the cache, the FLOPs
  per step are small, so total FLOPs are of the same order. **Latency** is not: d2tok has about 7
  times as many sequential steps as d2s. Time the test eval per model to put a number on it.
- **What d2tok gets for free:** each coordinate is its own classification, conditioned on the
  coordinates before it. A d2s node head predicts a node's coordinates side by side, not one after
  the other. Some of d2tok's lead may come from this finer factorisation, not from the order.

## Reproduce

```
uv run --with matplotlib --with pandas python plot_vitruvion.py
uv run --with matplotlib --with pandas --with numpy python analyse_vitruvion_errors.py
uv run --with matplotlib --with numpy python draw_transcripts.py <out> results/runs/vitruvion-d2tok-random-seed0 results/runs/vitruvion-final-d2po --disagree
```
