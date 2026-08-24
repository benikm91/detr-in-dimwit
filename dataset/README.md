# dataset

The [`benikm91/l-shape`](https://huggingface.co/datasets/benikm91/l-shape) drawings, in the two
views the models in this repository need.

## The record

A drawing is rendered from a **record**: the graph the drawing program spells out. That is the
source truth, and it is what `LShapeDataset` hands out.

```scala
val data = LShapeDataset.open(Axis[Width], Axis[Height], Axis[Channel], Axis[Node])(Split.Train)
data.samples                               // every drawing and its record, once
data.batches(Axis[Drawings] -> 32)         // batches, for as long as they are asked for
data.objects                               // the same drawings as something to detect
data.objectBatches(Axis[Drawings] -> 32)
```

`batches` never runs out: it cycles the split. `samples` is one finite pass, which is what
evaluation wants. Neither shuffles — the drawings were generated independently of one another, so
reading them in order already is one.

The records are read into tensors when a split is opened. The drawings stay memory mapped, since
the train split is 8.6 GB and only the batches asked for are ever read; Python hands over rows and
Scala does the rest.

`Line` and `Annotation` are drawn; `Connected` and `Annotates` are the relationships between them,
held as nodes of their own that name the nodes they link by where those sit — which is what makes
a graph a set, and is why a model that can predict a set can predict a graph. A node carries only
what its class carries: a line two points, a relationship two links.

Nothing in a record has an order. [`Record`](src/main/scala/Records.scala) is one *layout* of it
along a node axis, for the device — `nodeClass`, `xs`, `ys`, `links`;
[`RecordGraph`](src/main/scala/Records.scala) is the record itself, on the host, and going between
the two is how a record is permuted, compared or written down.

## The two views, and the one function between them

```
LShapeDataset.samples ─── Record ──Objects.of──▶ Objects ─── detr, egtr
                            │                       │
                            └───────────────────────┘
                                 RecordGraph.of
```

[`Objects.of`](src/main/scala/Objects.scala) is the whole difference between the record dataset
and the detection dataset: a box around every drawn node — a line spans its two end points, an
annotation is a fixed square around its point — and the relationships as an adjacency matrix over
the positions those boxes sit in, a symmetric one held both ways round.

`RecordGraph.of` goes back: a line is axis aligned, so the long side of its box is the line and
the short side is what `Objects.of` widened it to. That is what lets a detector's *prediction* be
compared with the record a drawing was actually rendered from rather than with the boxes derived
from it — [`ObjectsSuite`](src/test/scala/ObjectsSuite.scala) checks the round trip holds on the
ground truth of both splits.

Packing an adjacency matrix back into nodes is a compaction rather than an elementwise map, so
that direction reads the tensors to the host, which is where it is wanted — nothing scores on the
device.

## Scoring

[`RecordScoring`](src/main/scala/RecordScoring.scala) compares a predicted record with a target
one: nodes are matched by what they say, relationships by the nodes they name. It lives here
rather than in a model so that a detector, a scene graph model and a transcription model report
the same numbers.

## Files

| | |
|---|---|
| [Records.scala](src/main/scala/Records.scala) | `NodeClass`, `Record`, `RecordGraph`, and laying one out along the node axis |
| [Objects.scala](src/main/scala/Objects.scala) | `Objects.of`, and the boxes and adjacency it draws |
| [LShapeDataset.scala](src/main/scala/LShapeDataset.scala) | the loader and its two views |
| [RecordScoring.scala](src/main/scala/RecordScoring.scala) | comparing two records |
| [Box.scala](src/main/scala/Box.scala) | box geometry: L1 and GIoU |
| [Outlines.scala](src/main/scala/Outlines.scala) | drawing boxes over the drawing they came from |
| [python/l_shape_dataset.py](src/main/resources/python/l_shape_dataset.py) | parsing the drawing programs into records |
