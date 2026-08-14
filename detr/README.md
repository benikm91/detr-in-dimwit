# DETR

[End-to-End Object Detection with Transformers](https://arxiv.org/abs/2005.12872) on the
`benikm91/l-shape` drawings, built from deepwit modules. `DETR` maps one image to one
prediction — there is no batch axis in the architecture, batching is `vmap` at the training
level.

```
sbt "detr/runMain detrTrain"
```

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

**Matching runs outside the traced step.** The assignment is a discrete decision on concrete
values, so it cannot live inside `grad` or `jit`. `HungarianLoss.matchTargets` runs on its
own forward pass and hands the matched targets to the differentiable `HungarianLoss.apply`
as constants, which costs one extra forward pass per step. The solver is a shortest
augmenting path implementation ([HungarianMatching.scala](src/main/scala/HungarianMatching.scala))
rather than SciPy's `linear_sum_assignment`.

**Training setup.** The paper uses AdamW at 1e-4 (1e-5 for the backbone), weight decay 1e-4,
gradients clipped at 0.1, a step schedule over 300 epochs and scale/crop augmentation. This
trains with plain Adam at 1e-4, no schedule, no clipping and no augmentation — the dataset
generator already randomizes translation, mirroring and rotation.

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
| [Boxes.scala](src/main/scala/Boxes.scala) | box geometry: L1 and GIoU |
| [HungarianMatching.scala](src/main/scala/HungarianMatching.scala) | linear assignment |
| [HungarianLoss.scala](src/main/scala/HungarianLoss.scala) | matching and set prediction loss |
| [DETRTrain.scala](src/main/scala/DETRTrain.scala) | training loop |
