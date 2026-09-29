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
on, while `DETR.apply` settles every query on a node of the [record](../dataset), or on
`NoNode`, so targets and predictions render — and are scored — through the same code.

`detrEval` compares the nodes the queries answer with to the record the drawing was rendered from — a detector predicts no relationships, so the
record it is held against holds none either. It reports recall (`nodes detected`), precision
(`detections right`) and `drawings fully detected`, which is every node of a drawing found at once
with nothing spurious. Those are the node lines [`d2gEval`](../d2g/README.md) reports, on the same
records, so the two can be read against each other.

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

**The matcher is optimal, and runs on the device.** The paper solves the assignment with SciPy's
`linear_sum_assignment` on the host; [Matching.scala](src/main/scala/train/Matching.scala) calls
Optax's Hungarian algorithm instead, which is written in JAX and so traces, jits and vmaps with
the rest of the step.

**Training setup.** The paper uses AdamW at 1e-4 (1e-5 for the backbone), weight decay 1e-4,
gradients clipped at 0.1, a step schedule over 300 epochs and scale/crop augmentation. This
trains with plain Adam at 3e-4 — the model is far smaller and gets far fewer steps — with no
schedule, no clipping and no augmentation, since the dataset generator already randomizes
translation, mirroring and rotation.

**Nodes instead of boxes.** Every query predicts a node of the record the way the transcriber
does ([NodeHead.scala](src/main/scala/model/NodeHead.scala), copied from [d2g](../d2g)): a class
out of `NodeClasses`, `NoNode` standing in for the paper's "no object", and for every point a
class places — start, end, middle — the pixel it falls on. What a query costs against a node,
for the matching and the loss alike, is the cross entropy of the node's class and of each of
those pixels. Only the nodes are matched, so the queries left over are the ones trained towards
`NoNode`; the class term is averaged over all queries and the placement term over the nodes, as
the paper averages its class and box terms.

Unchanged from the paper: the encoder/decoder structure, learned object queries, and the set
prediction loss over an optimal matching.

## Files

| | |
|---|---|
| [DETRVocabulary.scala](src/main/scala/DETRVocabulary.scala) | axis labels, and coordinates as pixels |
| [DETR.scala](src/main/scala/model/DETR.scala) | the model and its parameters |
| [NodeHead.scala](src/main/scala/model/NodeHead.scala) | a node of the record out of every query |
| [Matching.scala](src/main/scala/train/Matching.scala) | optimal assignment, by Optax's Hungarian algorithm |
| [HungarianLoss.scala](src/main/scala/train/HungarianLoss.scala) | matching and set prediction loss |
| [DETRTrain.scala](src/main/scala/train/DETRTrain.scala) | training loop and checkpointing |
| [DETREval.scala](src/main/scala/eval/DETREval.scala) | plots and scores a checkpoint |

The [egtr](../egtr) module was built on the box detector this one used to be, and does not
build against it.
