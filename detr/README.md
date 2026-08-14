# DETR

[End-to-End Object Detection with Transformers](https://arxiv.org/abs/2005.12872) on the
`benikm91/l-shape` drawings, built from deepwit modules. `DETR` maps one image to one
prediction — there is no batch axis in the architecture, batching is `vmap` at the training
level.

```
sbt "detr/runMain detrTrain"           # trains, checkpointing every 500 steps into out/detr/<timestamp>
sbt "detr/runMain detrPlot"            # plots the first drawings of both splits with targets and predictions
sbt "detr/runMain detrEval"            # scores the newest checkpoint on the whole validation split
sbt "detr/runMain detrEval out/detr/…" # … or scores a given run
```

A checkpoint holds the whole `TrainState`, so training can be resumed from it and both eval
scripts read the parameters back out of it. `DETR.logits` gives the raw scores the loss works
on, while `DETR.apply` decides a class per query and returns the same `Detection` type the
dataset yields, so targets and predictions render — and are scored — through the same code.

`detrEval` matches predictions to targets exactly as training does, then counts:

- an **object** as detected when its class is right and its defining points land within 4
  pixels — both end points for a part line, the anchor for a text (the target's box size
  around a text is an artifact of the dataset wrapper, so it is not scored);
- a **drawing** as detected when every query slot is right: every object detected, none
  missing and none spurious.

Note that `detrPlot` reads the training split, which downloads 8.6 GB on first use.

## Divergences from the paper

**Vision transformer instead of a convolutional backbone.** The paper runs an
ImageNet-pretrained ResNet-50 with frozen BatchNorm and projects its `H/32 × W/32` feature
map to `d = 256` with a 1×1 convolution. The drawings here are synthetic single channel line
art, so a pretrained natural image backbone buys nothing: `ImageToPatchEmbedder` embeds
16×16 patches of the 256×256 canvas directly into 256 tokens. As a result the model contains
no BatchNorm at all — the only normalization is the LayerNorm inside the transformer blocks.

**Positional information is added once.** The paper adds fixed sine encodings to the queries
and keys of *every* encoder layer, and the learned object queries to those of every decoder
layer. Here the 2D sine encoding is added once by the patch embedder, and the learned object
queries are the decoder's initial input rather than a positional term re-added per layer.
deepwit's attention takes no separate positional argument.

**No auxiliary decoding losses.** The paper supervises the output of every decoder layer
through shared heads. Only the final layer is supervised here.
`CrossTransformer.applyWithHiddenStates` exposes the per-layer states, so this can be added.

**No dropout.** The paper uses 0.1 throughout the transformer; deepwit has no dropout module.

**The matcher is greedy, not optimal.** The paper solves the assignment exactly with SciPy's
`linear_sum_assignment`. An augmenting path algorithm needs data dependent control flow,
which would force the matching out of the traced step and onto the host; instead
[Matching.scala](src/main/scala/Matching.scala) repeatedly takes the cheapest remaining pair,
which is a fixed number of steps of plain tensor operations and therefore traces, jits and
vmaps with everything else. Greedy can be beaten on an ambiguous cost matrix, but keeping the
matching inside the graph made training about ten times faster, since nothing has to leave
the device mid-step. Padding slots carry a surcharge derived from the spread of the real
costs — a constant per column leaves the optimal assignment untouched, and it stops greedy
from handing its cheapest predictions to slots holding no object.

**Training setup.** The paper uses AdamW at 1e-4 (1e-5 for the backbone), weight decay 1e-4,
gradients clipped at 0.1, a step schedule over 300 epochs and scale/crop augmentation. This
trains with plain Adam at 3e-4 — the model is far smaller and gets far fewer steps — with no
schedule, no clipping and no augmentation, since the dataset generator already randomizes
translation, mirroring and rotation.

**Task.** Three classes (`NoObject`, `PartLine`, `Text`) instead of 91 COCO classes, and the
targets come padded to the query count, which is how the paper pads ground truth to `N`, so
the assignment is square. Boxes are normalized `(cx, cy, w, h)` from a sigmoid head and the
loss is the paper's: cross entropy with the "no object" class down-weighted by 0.1, plus L1
and GIoU on matched pairs, weighted 1 / 5 / 2.

Unchanged from the paper: the encoder/decoder structure, learned object queries, the linear
class head, the three layer perceptron box head, and the set prediction loss.

## Files

| | |
|---|---|
| [Vocabulary.scala](src/main/scala/Vocabulary.scala) | axis labels shared by the model and the dataset |
| [DETR.scala](src/main/scala/DETR.scala) | the model and its parameters |
| [Matching.scala](src/main/scala/Matching.scala) | greedy assignment, in tensor operations |
| [HungarianLoss.scala](src/main/scala/HungarianLoss.scala) | matching and set prediction loss |
| [DETRTrain.scala](src/main/scala/DETRTrain.scala) | training loop and checkpointing |
| [DETREval.scala](src/main/scala/DETREval.scala) | plots and scores a checkpoint |

Box geometry (L1 and GIoU) lives with `Detection` in the dataset module,
[Box.scala](../dataset/src/main/scala/Box.scala).
