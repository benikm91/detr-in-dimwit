import d2tok.config.D2TokModelConfiguration
import dataset.Corpus
import dataset.DrawingDataset.Split
import syntheticFloorplansDataset.FloorplanCorpus

/** Entry points for the D2Tok experiments, on a corpus (e.g., `vitruvion-primitives` or
  * `synthetic-floorplans`) at a model configuration (e.g., `s-deep`):
  * `sbt "d2tok/runMain d2tokTrain vitruvion-primitives s-deep random 0"`.
  *
  * Each kind of record has a language of its own: the sketch corpora (see [[Corpus]]) are written by
  * [[d2tok.sketch]], the synthetic floor plans by [[d2tok.floorplan]]. The corpus names which, and
  * the order is one of that language's: [[d2tok.sketch.SequenceOrder]] or
  * [[d2tok.floorplan.SequenceOrder]]. See [[D2TokModelConfiguration]] for the sizes.
  *
  * [[d2tokTrain]] trains a run in an order, from a seed.
  * [[d2tokEval]] scores the newest run of that corpus and size (greedy decoding).
  * [[d2tokTest]] scores its last checkpoint on the test split.
  * [[d2tokDraw]] writes what every checkpoint transcribed as a picture.
  *
  * The `d2tokGraph` entry points do the same for records with their relationships, written after the
  * nodes in an order of their own, in a single sequence: the floor plans by
  * [[d2tok.floorplangraph]], the drawing corpora that hold relationships by [[d2tok.sketchgraph]]:
  * `sbt "d2tok/runMain d2tokGraphTrain rectilinear s-deep class by-kind 0"`.
  * [[d2tokGraphTrainInSlices]] trains on floor plans for a batch too big for the GPUs, a slice of it
  * at a time.
  */

@main
def d2tokTrain(corpus: String, size: String, order: String, seed: Int): Unit = floorplans(corpus) match
  case Some(plans) =>
    val setup = d2tok.floorplan.config.D2TokSetup(plans, D2TokModelConfiguration.named(size))
    d2tok.floorplan.train.trainWordTranscriber(setup.copy(order = d2tok.floorplan.SequenceOrder.named(order), seed = seed))
  case None =>
    val setup = d2tok.sketch.config.D2TokSetup(Corpus.named(corpus), D2TokModelConfiguration.named(size))
    d2tok.sketch.train.trainWordTranscriber(setup.copy(order = d2tok.sketch.SequenceOrder.named(order), seed = seed))

@main
def d2tokEval(corpus: String, size: String): Unit = score(corpus, size, Split.Validation)

@main
def d2tokTest(corpus: String, size: String): Unit = score(corpus, size, Split.Test)

@main
def d2tokDraw(corpus: String, size: String): Unit = floorplans(corpus) match
  case Some(plans) => d2tok.floorplan.eval.drawWordTranscriber(d2tok.floorplan.config.D2TokSetup(plans, D2TokModelConfiguration.named(size)), size)
  case None        => d2tok.sketch.eval.drawWordTranscriber(d2tok.sketch.config.D2TokSetup(Corpus.named(corpus), D2TokModelConfiguration.named(size)), size)

private def score(corpus: String, size: String, split: Split): Unit = floorplans(corpus) match
  case Some(plans) => d2tok.floorplan.eval.scoreWordTranscriber(d2tok.floorplan.config.D2TokSetup(plans, D2TokModelConfiguration.named(size)), size, split)
  case None        => d2tok.sketch.eval.scoreWordTranscriber(d2tok.sketch.config.D2TokSetup(Corpus.named(corpus), D2TokModelConfiguration.named(size)), size, split)

/** The floor plan corpus a name stands for, if it stands for one rather than for a [[Corpus]] of sketches. */
private def floorplans(corpus: String): Option[FloorplanCorpus] = FloorplanCorpus.values.find(_.name == corpus)

@main
def d2tokGraphTrain(corpus: String, size: String, nodeOrder: String, relationshipOrder: String, seed: Int): Unit = floorplans(corpus) match
  case Some(plans) =>
    d2tok.floorplangraph.train.trainGraphTranscriber(floorplanGraphSetup(plans, size).copy(
      nodeOrder = d2tok.floorplan.SequenceOrder.named(nodeOrder),
      relationshipOrder = d2tok.floorplangraph.RelationshipOrder.named(relationshipOrder),
      seed = seed
    ))
  case None =>
    d2tok.sketchgraph.train.trainGraphTranscriber(sketchGraphSetup(corpus, size).copy(
      order = d2tok.sketch.SequenceOrder.named(nodeOrder),
      relationshipOrder = d2tok.sketchgraph.RelationshipOrder.named(relationshipOrder),
      seed = seed
    ))

@main
def d2tokGraphTrainInSlices(corpus: String, size: String, nodeOrder: String, relationshipOrder: String, seed: Int, slices: Int): Unit =
  d2tok.floorplangraph.train.trainGraphTranscriberInSlices(
    floorplanGraphSetup(FloorplanCorpus.named(corpus), size).copy(
      nodeOrder = d2tok.floorplan.SequenceOrder.named(nodeOrder),
      relationshipOrder = d2tok.floorplangraph.RelationshipOrder.named(relationshipOrder),
      seed = seed
    ),
    slices
  )

@main
def d2tokGraphEval(corpus: String, size: String): Unit = scoreGraph(corpus, size, Split.Validation)

@main
def d2tokGraphTest(corpus: String, size: String): Unit = scoreGraph(corpus, size, Split.Test)

@main
def d2tokGraphDraw(corpus: String, size: String): Unit =
  d2tok.floorplangraph.eval.drawGraphTranscriber(floorplanGraphSetup(FloorplanCorpus.named(corpus), size), size)

private def scoreGraph(corpus: String, size: String, split: Split): Unit = floorplans(corpus) match
  case Some(plans) => d2tok.floorplangraph.eval.scoreGraphTranscriber(floorplanGraphSetup(plans, size), size, split)
  case None        => d2tok.sketchgraph.eval.scoreGraphTranscriber(sketchGraphSetup(corpus, size), size, split)

private def floorplanGraphSetup(plans: FloorplanCorpus, size: String) = d2tok.floorplangraph.config.D2TokGraphSetup(plans, D2TokModelConfiguration.named(size))

private def sketchGraphSetup(corpus: String, size: String) = d2tok.sketchgraph.config.D2TokGraphSetup(Corpus.named(corpus), D2TokModelConfiguration.named(size))
