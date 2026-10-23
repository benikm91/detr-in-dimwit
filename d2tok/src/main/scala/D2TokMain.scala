import d2tok.config.D2TokModelConfiguration
import d2tok.floorplangraph.config.D2TokGraphSetup
import dataset.Corpus
import dataset.DrawingDataset.Split
import syntheticFloorplansDataset.SyntheticFloorplans

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
  * The `d2tokGraph` entry points do the same for the floor plans with their relationships, written
  * by [[d2tok.floorplangraph]] after the nodes, in an order of their own:
  * `sbt "d2tok/runMain d2tokGraphTrain s-deep walls-first by-nodes 0"`.
  */

@main
def d2tokTrain(corpus: String, size: String, order: String, seed: Int): Unit =
  if corpus == SyntheticFloorplans.name then
    val setup = d2tok.floorplan.config.D2TokSetup(D2TokModelConfiguration.named(size))
    d2tok.floorplan.train.trainWordTranscriber(setup.copy(order = d2tok.floorplan.SequenceOrder.named(order), seed = seed))
  else
    val setup = d2tok.sketch.config.D2TokSetup(Corpus.named(corpus), D2TokModelConfiguration.named(size))
    d2tok.sketch.train.trainWordTranscriber(setup.copy(order = d2tok.sketch.SequenceOrder.named(order), seed = seed))

@main
def d2tokEval(corpus: String, size: String): Unit = score(corpus, size, Split.Validation)

@main
def d2tokTest(corpus: String, size: String): Unit = score(corpus, size, Split.Test)

@main
def d2tokDraw(corpus: String, size: String): Unit =
  if corpus == SyntheticFloorplans.name then d2tok.floorplan.eval.drawWordTranscriber(d2tok.floorplan.config.D2TokSetup(D2TokModelConfiguration.named(size)), size)
  else d2tok.sketch.eval.drawWordTranscriber(d2tok.sketch.config.D2TokSetup(Corpus.named(corpus), D2TokModelConfiguration.named(size)), size)

private def score(corpus: String, size: String, split: Split): Unit =
  if corpus == SyntheticFloorplans.name then
    d2tok.floorplan.eval.scoreWordTranscriber(d2tok.floorplan.config.D2TokSetup(D2TokModelConfiguration.named(size)), size, split)
  else
    d2tok.sketch.eval.scoreWordTranscriber(d2tok.sketch.config.D2TokSetup(Corpus.named(corpus), D2TokModelConfiguration.named(size)), size, split)

@main
def d2tokGraphTrain(size: String, nodeOrder: String, relationshipOrder: String, seed: Int): Unit =
  val setup = D2TokGraphSetup(D2TokModelConfiguration.named(size))
  d2tok.floorplangraph.train.trainGraphTranscriber(setup.copy(
    nodeOrder = d2tok.floorplan.SequenceOrder.named(nodeOrder),
    relationshipOrder = d2tok.floorplangraph.RelationshipOrder.named(relationshipOrder),
    seed = seed
  ))

@main
def d2tokGraphEval(size: String): Unit = d2tok.floorplangraph.eval.scoreGraphTranscriber(D2TokGraphSetup(D2TokModelConfiguration.named(size)), size, Split.Validation)

@main
def d2tokGraphTest(size: String): Unit = d2tok.floorplangraph.eval.scoreGraphTranscriber(D2TokGraphSetup(D2TokModelConfiguration.named(size)), size, Split.Test)

@main
def d2tokGraphDraw(size: String): Unit = d2tok.floorplangraph.eval.drawGraphTranscriber(D2TokGraphSetup(D2TokModelConfiguration.named(size)), size)
