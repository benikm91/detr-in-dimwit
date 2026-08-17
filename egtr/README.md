# EGTR

[Extracting Graph from Transformer for Scene Graph Generation](https://arxiv.org/abs/2404.02072)
on the `benikm91/l-shape` drawings, built on the [DETR](../detr) of this repository — the
detector is a dependency, not a copy, and nothing in `detr` knows about this module.

A scene graph is a detection plus the edges between what was detected. EGTR's point is that a
DETR decoder has already related its object queries to each other: to decide what it stands for,
every query reads every other one, and the queries and keys it reads them by are a statement
about the pair. So the edges are extracted from that by-product by a shallow head, rather than
predicted by relation queries or a second decoder.

```
sbt "egtr/runMain egtrTrain"                 # trains, from the newest detrTrain run if there is one
sbt "egtr/runMain egtrTrain ../detr/out/detr/…"  # … or from a given detector
sbt "egtr/runMain egtrEval"                  # scores the newest checkpoint on the whole validation split
sbt "egtr/runMain egtrEval out/egtr/…"       # … or scores a given run
```

Note that sbt forks each `runMain` from its own project directory, so these runs live in
`egtr/out/egtr` while the detector's live in `detr/out/detr`.

## The graph of a drawing

The nodes are what `detr` already detects: the `PartLine`s of the outline and the `Text` of a
dimension annotation. The edges are the relationship actions of the drawing program, which the
dataset wrapper used to drop and now parses:

| | |
|---|---|
| `Connected` | two part lines meeting in a corner of the outline. Six per drawing, and symmetric — which of the two the drawing program names first carries no meaning, so it is stored and predicted both ways round. |
| `Annotates` | the part line whose length a dimension annotation measures. Zero to six per drawing, and directed: `Text → PartLine`. |

Unlike `ObjectClass` there is no "no relation" class. A pair of objects carries each relation or
it does not, independently of the others, so the head is a set of binary decisions under a
sigmoid rather than a classification under a softmax, and an unrelated pair is simply zero
everywhere. Both splits parse with no reference left unresolved
(`LShapeDetectionDataset.unresolvedRelations`), so every edge of every drawing is between two
objects that are detected.

## What `egtrEval` counts

Nodes are matched and scored exactly as [`detrEval`](../detr/README.md) scores them, so the
object lines mean the same thing in both. Edges are then scored *on top of* the nodes: a target
edge is only credited when both of its objects were detected, since an edge between misplaced
boxes relates nothing.

The graph of a drawing is 32 × 32 × 2 = 2048 entries of which some 14 are edges, so most of what
is reported is the *ranking* of the triplets rather than a decision taken at some cut-off — as in
the paper, and for the same reason: a ranking is what a model this sparse can be compared on
without a threshold having to be argued for.

- **`relations R@k`** — of the target edges, how many are among the `k` best scoring triplets of
  their drawing. `connected` and `annotates` break the same ranking down per relation, which
  matters because the two are neither equally frequent nor equally hard.
- **`drawings fully correct`** — every query slot right *and* every target edge outranking every
  triplet that is not one. The whole graph of a drawing in one number, with no threshold to pick.
- **`relations over 0.5`** — of the target edges, how many actually score above one half. Unlike
  the lines above, this one asks the scores to be calibrated and not merely ordered, which is the
  question a consumer of the graph really has. It tracks `R@20` closely here, a few points below
  it, so the scores are ordered *and* calibrated rather than only ordered.

## Divergences from the paper

**Plain DETR instead of Deformable DETR.** The paper builds on Deformable DETR with a ResNet-50
and 200 object queries, and notes the approach extends to any detector whose object features
attend to each other. Here it is the DETR of this repository: a vision transformer backbone, 3
decoder blocks, 4 heads, 128 wide embeddings and 32 queries — which also means the graph is 2048
entries rather than two million, and that is what makes the negative sampling below unnecessary.

**The self-attention by-products are intercepted, not recomputed.** Equation 3 reads the queries
and keys of each decoder block's self-attention. deepwit's attention hands them out —
`MultiHeadAttention.applyWithIntermediates` answers with the projections next to the attended
target — so
[`DETRDecoder.applyWithSelfAttentionIntermediates`](../detr/src/main/scala/DETRDecoder.scala)
carries them up the block stack rather than projecting a second time. They are by construction
the ones the attention attended by, and cost nothing beyond the detection itself. What the block
does pay for is the residual skeleton it repeats to leave them a way out, deepwit's
`CrossTransformerBlock` having none.

**Region weights instead of hard negative sampling.** §3.3.1 splits the graph into the target
edges, the negatives (pairs of detected objects carrying no edge) and the non-matching region
(pairs reaching into the padding), and samples the `k_neg × |E|` hardest of the latter two, with
`k = 80`. That exists because Visual Genome's graph is 200 × 200 × 50 entries at a density of
1e-14; here the density is 0.7%, and at `80 × 14` the sampling would keep more entries than the
graph has. So the three regions are weighted instead — `1`, `negativeWeight`,
`nonMatchingWeight` — which needs no sort, no top-k and no count that depends on the data inside
the traced step. The default weights are `1` and `0.1`: a pair of real objects without an edge is
the informative negative, a pair reaching into the padding is neither informative nor rare.

**`α` is set for this cost scale.** In equation 8 the uncertainty of a detected object is
`σ(cost − cost_min + σ⁻¹(α))`, and the paper sets `α = 1e-14`, i.e. `σ⁻¹(α) ≈ −32`. Matching
costs here span roughly `[−1, 10]`, so smoothing would never engage at that value. `α = 0.02`
puts `σ⁻¹(α) ≈ −3.9`, which makes a query costing about four more than a perfect match half
uncertain and halves the relations it takes part in. `α` is therefore not only the floor of the
uncertainty but the scale of the whole smoothing.

**The smoothing is a label, so no gradient flows through it.** Equation 8 measures a detected
object by its matching cost, which here is a differentiable quantity of the model rather than
something a `no_grad` matcher handed over as in the paper's implementation. Left differentiable it
is a shortcut: a query uncertain enough has every relation it takes part in damped towards zero,
so the cheapest way to satisfy the relation loss is to detect badly. That is not a hypothetical —
before the cost was detached, this model's detection fell from 98.8% of objects to 48.9% over the
first thousand steps while the relation loss went quiet, and `Connected`, whose two ends are both
damped, was never learned at all. dimwit has no stop-gradient, so `EGTRLoss` reaches for
`jax.lax.stop_gradient`.

**The connectivity target is smoothed too.** §3.3.2 builds the connectivity graph "in a similar
way" to the relation graph; here that is taken literally, and a pair's connectivity target is
damped by the uncertainty of its two objects exactly as its relations are.

**No logit adjustment.** The paper adjusts logits for the tail predicates of Visual Genome's 50.
There are two predicates here and both are common.

**Training setup.** One optimizer over the whole model, as the paper does after its detector
pre-training stage: AdamW at 3e-4, weight decay 1e-4, gradients clipped at 0.1, batches of 32.
Starting from a trained detector is supported and is the default, but the detector is not frozen
— it keeps training on the joint loss.

Unchanged from the paper: the pairwise concatenation of every source (eq. 3, 4), the per-source
gate and the gated sum (eq. 5, 6), the two three-layer perceptron heads, the connectivity task
as an auxiliary loss, adaptive smoothing of the relation labels by the matching cost (eq. 8), the
multi-task loss with `λ_rel = 15` and `λ_con = 30` (eq. 7), and the inference of §3.3.3 —
predicate score times both class scores times connectivity, with the diagonal cleared so that
nothing relates to itself.

## Files

| | |
|---|---|
| [EGTRVocabulary.scala](src/main/scala/EGTRVocabulary.scala) | the axis labels the graph runs over |
| [SceneGraph.scala](src/main/scala/SceneGraph.scala) | objects plus the grid of relations between them, for targets and predictions alike |
| [RelationExtractor.scala](src/main/scala/RelationExtractor.scala) | equations 3 to 6: the sources, the gated sum and the two heads |
| [EGTR.scala](src/main/scala/EGTR.scala) | the model, its parameters, and the inference of §3.3.3 |
| [EGTRLoss.scala](src/main/scala/EGTRLoss.scala) | equation 7: the detection loss, the relations and the connectivity |
| [EGTRTrain.scala](src/main/scala/EGTRTrain.scala) | training loop and checkpointing |
| [EGTREval.scala](src/main/scala/EGTREval.scala) | scores a checkpoint on the validation split |

What is reused from `detr` rather than rebuilt: the detector and its parameters, the greedy
matcher, and the set prediction loss — `HungarianLoss.matched` exposes the assignment and its
per-query cost, which is what the graph is permuted by and what equation 8 measures uncertainty
from. Node scoring is shared with `detrEval` through
[DetectionScoring.scala](../detr/src/main/scala/DetectionScoring.scala).
