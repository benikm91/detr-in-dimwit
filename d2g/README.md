# D2G — document-to-graph transcription

[A document is worth a structured record: Principled inductive bias design for document
recognition](https://arxiv.org/abs/2507.08458) on the `benikm91/l-shape` drawings, built from
deepwit modules. The paper's §3.7 *graph bias* — remaining-node prediction over an implicit graph
— is what this module implements; see the divergences below for what is done differently.

```
sbt "d2g/runMain d2gTrain"          # trains, checkpointing every 1000 steps into out/d2g/<timestamp>
sbt "d2g/runMain d2gPlot"           # plots the first drawings of both splits with target and transcribed records
sbt "d2g/runMain d2gEval"           # scores the newest checkpoint on the whole validation split
sbt "d2g/runMain d2gEval out/d2g/…" # … or scores a given run
```

## The idea

[detr](../detr) and [egtr](../egtr) treat the drawing as an object detection problem and recover
the structure afterwards. This model does not detect anything: it **transcribes the drawing into
its record** — the graph the drawing was rendered from — one node at a time. A part line is
predicted as a line, not as a box around a line, and a corner is predicted as a node of the record
that names the two lines meeting in it.

**The record is a set of nodes.** A node has a type and discrete properties. Edges are nodes too:
the paper represents a graph relationally, so an undirected edge between `A` and `B` becomes a
*relationship node* `Connected(A, B)` whose links are the two nodes it
links. A drawing's record is therefore just `{Line, …, Annotation, …, Connected(…), Annotates(…)}`
— a set, with no order in it, which is exactly why next-node prediction is the wrong bias for it.

**Remaining-node prediction instead of next-node prediction.** A set has no next element, so the
decoder is not asked for the next node but for *any node it has not taken yet*. Training feeds it
a random permutation of the record's nodes and, at every position, asks for one of the nodes from
that position onwards — the loss is the **minimum** dissimilarity over all remaining nodes
(Eq. 4b/4c of the paper), so any of them is an acceptable answer. Inference is then the ordinary
autoregressive loop: take what was predicted, append it, ask again, stop at `<EOS>`.

**Two embeddings per node, because an embedding has two jobs.** Teacher forcing wants every
position predicted in one pass. But a node embedding that is transformed into its prediction no
longer carries the node it stands for, and the later positions need it to know what is still
remaining. So the decoder holds two embeddings per node: a **node embedding**, which carries the
taken node, and a **prediction embedding**, which becomes one of the remaining nodes. The
prediction embeddings are never attended to — they hold what the model is guessing, which is of no
use to anyone else.

**The drawn nodes come first.** A relationship names other nodes, so it can only be predicted once
they exist. Permutations are therefore drawn within two blocks — drawn nodes, then relationships —
and a node is named by where it sits, which the decoder reads off the positional encoding. Predicting the record is thus symbol recognition
followed by symbol assembly, without either being a separate stage.

## The decoder sequence

The paper interleaves the two embeddings, `<P> n₁ <P> n₂ …`. Here they are **appended** instead:

```
node        0    1    2         N-1  |  0    1    2         N-1
        [  n₀   n₁   n₂   …   pad     |  p₀   p₁   p₂   …   pad  ]
           node embeddings            |  prediction embeddings
```

Where an embedding lives is not a fact the attention has access to — only the mask and the
positional encoding are — so the layout is free, and this one costs nothing and reads better:
node `i` is at `i`, the prediction embedding that answers for it is at `i` of the second half, and
both carry the positional encoding of `i`, which is what a relationship names it by.

The mask says the rest. Writing `i` and `j` for the position within a half:

| row (attends) | column (attended) | allowed when |
|---|---|---|
| anything | itself | always |
| node `i` | node `j` | `j ≤ i` and `j` holds a node |
| prediction `i` | node `j` | `j < i` and `j` holds a node |
| anything | any other prediction | never |

The node embeddings are causal over the taken nodes and include themselves — that is the
pass-through role. The prediction embeddings see the *taken* nodes strictly before them and
nothing else, so what they may predict is exactly what is left. Nothing reads a prediction
embedding, which carries a guess rather than a fact. The diagonal is what keeps the first
prediction embedding — and every position of a record with nothing taken yet — from being a fully
masked row, which would have no softmax.

## The record of an l-shape drawing

The record is what [the dataset](../dataset) hands out — the graph the drawing program spells
out, and not something read back off boxes:

| node class | carries | |
|---|---|---|
| `Line` | two points | a segment of the outline, by its two end points |
| `Annotation` | one point | a dimension annotation, by the point it is centred on |
| `Connected` | two links | two lines meeting in a corner |
| `Annotates` | two links | an annotation and the line it measures |
| `NoNode` | — | nothing, which is also where a transcription stops |

A point is normalized to the canvas and this model predicts its `x` and its `y` as **the pixel
each falls on**; a link is predicted as **the node it names**. Both are one discrete choice, which
is the only kind of property here.

`NoNode` doing double duty is deliberate: a position the record does not reach and the answer
"the record has ended" are the same statement, so the paper's `<EOS>` is not a symbol of its own.

## What `d2gEval` counts

Transcription is scored as the paper scores it: **all or nothing per drawing**, up to a coordinate
tolerance. A predicted record counts only if it is the target record — same nodes, same
coordinates, same links, none missing and none spurious. Records are compared as records rather
than slot by slot, so relationships are compared by the *nodes they name* and a transcription that
emits the same graph in a different order is correct.

Because that number is unforgiving, the parts are reported too — how many nodes and how many
relationships of the split were transcribed, and how many drawings stop at the right length. The
node lines are what [`detrEval`](../detr/README.md) reports, on the same records, so the two can
be read against each other.

Drawings are transcribed in batches and in lockstep — every drawing of a batch takes its first
node, then its second, and one that has already stopped takes nothing more — so where a
transcription ends is a value rather than a branch, and the whole of it is a single compiled
computation per batch. What a drawing ends up with is the record it would have been given on its
own.

Inference still runs without a KV cache: every step re-reads the whole sequence, which makes the
model no different and the graph larger than it needs to be. That graph is now what the evaluation
mostly costs — it is compiled once, and the split itself takes seconds.

## Divergences from the paper

**Prediction embeddings are appended, not interleaved.** See above.

**Cross-attention is cross-attention.** The reference implementation concatenates the encoded
patches onto the decoder sequence and runs one attention over the concatenation, so the image and
the record compete in a single softmax and share one key and one value projection. Here the
decoder block is the ordinary one: masked self-attention over the record, then cross-attention
onto the encoded document, then the MLP, each on its own pre-normed residual branch — the same
block [detr](../detr) uses, differing only in the mask.

**The candidate set follows the equations, not the reference code.** In the reference
implementation the minimum of Eq. 4b is taken only over the remaining nodes *whose type equals the
type of the ground-truth node at that position*. That puts the permutation's type order back into
a loss whose whole point is to have no order: the model is effectively told which type to emit
next. Here the candidates are all remaining nodes of the same block, as Eq. 4b and 4c say.

**Property losses are not gated on the predicted type.** §4.5 lets a property loss count only if
the node type was predicted correctly. That gate is discontinuous in the model's own output and
starves the property heads of gradient exactly while the type head is still wrong. Here a property
is supervised whenever the *target* node has one, which depends on the data and not on the
prediction.

**Discrete properties only.** The paper predicts a coordinate coarse-to-fine — a patch by
cross-attention, then a pixel inside it — and also supports continuous properties. Here a
coordinate is one 256-way choice over pixel indices and a reference one choice over the decoder's
slots, and there is nothing else. It costs the sub-pixel resolution the paper does not have
either, and it removes a second attention pass from the head.

**A node carries only what its class carries.** The heads that a class does not use are
unsupervised, so what they say is cleared when a node is decided rather than fed back into the
decoder. Without that, a node the model takes carries values no node from the data ever would,
and the next slot cannot tell what has been taken.

**No empty-patch removal.** The paper drops the all-white patches of a drawing before encoding,
which is most of them. dimwit's shapes are static, so the encoder would have to mask rather than
drop, and it is not done here.

**The losses are averaged, not summed.** Eq. 4 sums over the record, which weights a drawing by
how much is in it. The remaining-node and `<EOS>` terms are divided by the number of predictions
they cover and the pass-through term by the number of nodes, so a drawing with six annotations
counts as much as one with none.

**Training setup.** AdamW at 3e-4, weight decay 1e-4, gradients clipped at 0.1, batches of 32 —
the setup [egtr](../egtr) uses, not the paper's 1e-4 AdamW over 40 epochs, since this model is far
smaller than the paper's 65 M parameters.

## Files

| | |
|---|---|
| [Vocabulary.scala](src/main/scala/Vocabulary.scala) | the axes this model predicts in, and the pixel a coordinate is |
| [RemainingNodePrediction.scala](src/main/scala/RemainingNodePrediction.scala) | the decoder sequence and the attention mask above — the inductive bias itself |
| [Transformer.scala](src/main/scala/Transformer.scala) | the document encoder and the record decoder it is applied in |
| [NodeEmbedding.scala](src/main/scala/NodeEmbedding.scala) | a record read into embeddings, and embeddings read back into a record |
| [D2G.scala](src/main/scala/D2G.scala) | the model, and the step transcription advances by |
| [RemainingNodeLoss.scala](src/main/scala/RemainingNodeLoss.scala) | equation 4: remaining nodes, remaining relationships, the stop, and pass-through |
| [D2GTrain.scala](src/main/scala/D2GTrain.scala) | training loop and checkpointing |
| [D2GEval.scala](src/main/scala/D2GEval.scala) | transcribes a split autoregressively and scores it, and plots what it transcribes |

The record itself, and scoring one against another, belong to [the dataset](../dataset) — which
is what lets `detrEval` report the same node lines.
