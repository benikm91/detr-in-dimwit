# dataset

The drawing corpora, the records they were rendered from, and how a predicted record is scored
against them. Every model in this repository reads its data and reports its numbers through here,
so that two models are compared by the same code and not merely by the same words.

## The corpora

A [`Corpus`](src/main/scala/DrawingDataset.scala) is a Hugging Face repository of drawings and
the labels they were rendered from:

- `l-shape`, `rectilinear`, `sketch` and `sketch-xl`: 256 × 256, labelled by drawing programs;
- `vitruvion-primitives`: 128 × 128, the SketchGraphs sketches as Vitruvion selected and rendered
  them, in their own splits, labelled by their primitives. The constraints the same files hold are
  not read.

A corpus carries where it is published, the side of its drawings, and how many nodes and
relationships its largest record holds.

```scala
val data = DrawingDataset.open(Corpus.SketchGraph)(Axis[Width], Axis[Height], Axis[Channel], Axis[Node], Axis[Edge])(Split.Train)
data.samples                           // every drawing and its record, once, on the device
data.batches(Axis[Drawings] -> 128)    // batches for as long as they are asked for, on the host
```

`batches` never runs out: it cycles the split. A batch stays on the host as the corpus stores it,
one byte a pixel, so that sharding it is its one transfer to the devices; `drawingsOf` turns its
pixels into what a model reads, inside the jitted step. `samples` is one finite pass, which is
what evaluation wants. Neither shuffles — the drawings were generated independently of one
another, or shuffled when the corpus was built, so reading them in order already is one.

## The record

A drawing is rendered from a **record**: the graph its drawing program spells out. A node is a
line, a circle, an arc, an annotation or a point, placed by the points its class names, and
construction geometry or not; a relationship links two nodes, `Connected` corners and `Annotates`
dimensions.

Nothing in a record has an order. [`Record`](src/main/scala/Records.scala) is one *layout* of it
along the node and edge axes, for the device; [`RecordGraph`](src/main/scala/Records.scala) is the
record itself, on the host, and going between the two is how a record is permuted, compared or
written down.

## Scoring

[`RecordScoring`](src/main/scala/RecordScoring.scala) compares a predicted record with the target
one at a pixel tolerance: nodes are matched by what they say, relationships by the nodes they
name. [`Metrics`](src/main/scala/Metrics.scala) writes the scores of every checkpoint as one CSV,
and [`Transcripts`](src/main/scala/Transcripts.scala) what the last checkpoint predicted for every
drawing, beside the target and the matching the scores were computed from.

## Files

| | |
|---|---|
| [DrawingDataset.scala](src/main/scala/DrawingDataset.scala) | `Corpus`, the loader, and turning stored pixels into what a model reads |
| [Records.scala](src/main/scala/Records.scala) | `NodeClass`, `EdgeClass`, `Record`, `RecordGraph`, and laying one out |
| [RecordScoring.scala](src/main/scala/RecordScoring.scala) | comparing two records |
| [Metrics.scala](src/main/scala/Metrics.scala) | a run's scores, one CSV row per checkpoint and tolerance |
| [Transcripts.scala](src/main/scala/Transcripts.scala) | what a model wrote down for every validation drawing |
| [RecordDrawing.scala](src/main/scala/RecordDrawing.scala) | drawing a record as the picture it stands for |
| [DrawingTiles.scala](src/main/scala/DrawingTiles.scala) | many drawings in one picture |
| [Runs.scala](src/main/scala/Runs.scala), [History.scala](src/main/scala/History.scala) | where a run keeps its checkpoints, outputs and loss history |
| [python/drawing_dataset.py](src/main/resources/python/drawing_dataset.py) | parsing the drawing programs into records |
