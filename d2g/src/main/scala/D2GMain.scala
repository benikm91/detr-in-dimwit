import d2g.floorplan.RelationshipOrder
import d2g.floorplan.config.D2GSetup
import d2g.floorplan.eval.*
import d2g.floorplan.train.*
import d2s.config.D2SModelConfiguration
import d2s.floorplan.NodeOrder
import dataset.DrawingDataset.Split
import syntheticFloorplansDataset.FloorplanCorpus

/** Entry points for the D2G experiments on a corpus of synthetic floor plans (see
  * [[FloorplanCorpus]]), at a model configuration (e.g., `s`):
  * `sbt "d2g/runMain d2gTrain synthetic-floorplans s free free 0"`. See [[D2SModelConfiguration]] for
  * the sizes, [[NodeOrder.named]] for the orders of the nodes and [[RelationshipOrder.named]] for
  * those of the relationships.
  *
  * [[d2gTrain]] trains a run in an order of the nodes and of the relationships, from a seed, so that a
  * setting can be run again.
  * [[d2gFineTune]] trains every part of such a run further, from one of its checkpoints.
  * [[d2gEval]] scores the newest run of that corpus and size (greedy decoding).
  * [[d2gTest]] scores its last checkpoint on the test split.
  * [[d2gPlot]] shows what it transcribes (visualization).
  */

@main
def d2gTrain(corpus: String, size: String, order: String, relationshipOrder: String, seed: Int): Unit =
  trainTranscriber(setup(corpus, size).copy(nodeOrder = NodeOrder.named(order), relationshipOrder = RelationshipOrder.named(relationshipOrder), seed = seed))

/** As long as training from scratch, at a learning rate that leaves what the model already knows
  * largely in place.
  */
@main
def d2gFineTune(corpus: String, size: String, order: String, relationshipOrder: String, seed: Int, checkpoint: String): Unit =
  fineTuneTranscriber(
    setup(corpus, size).copy(nodeOrder = NodeOrder.named(order), relationshipOrder = RelationshipOrder.named(relationshipOrder), seed = seed, learningRate = 1e-4f),
    checkpoint
  )

@main
def d2gEval(corpus: String, size: String): Unit = scoreTranscriber(setup(corpus, size), size, Split.Validation)

@main
def d2gTest(corpus: String, size: String): Unit = scoreTranscriber(setup(corpus, size), size, Split.Test)

@main
def d2gPlot(corpus: String, size: String): Unit = plotTranscriber(setup(corpus, size))

private def setup(corpus: String, size: String): D2GSetup = D2GSetup(FloorplanCorpus.named(corpus), D2SModelConfiguration.named(size))
