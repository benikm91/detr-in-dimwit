import d2g.floorplan.config.D2GSetup
import d2g.floorplan.eval.*
import d2g.floorplan.train.*
import d2s.config.D2SModelConfiguration
import d2s.floorplan.NodeOrder
import dataset.DrawingDataset.Split

/** Entry points for the D2G experiments on the synthetic floor plans, at a model configuration
  * (e.g., `s`): `sbt "d2g/runMain d2gTrain s walls-first 0"`. See [[D2SModelConfiguration]] for the
  * sizes and [[NodeOrder.named]] for the orders of the nodes.
  *
  * [[d2gTrain]] trains a run in an order of the nodes, from a seed, so that a setting can be run again.
  * [[d2gEval]] scores the newest run of that size (greedy decoding).
  * [[d2gTest]] scores its last checkpoint on the test split.
  * [[d2gPlot]] shows what it transcribes (visualization).
  */

@main
def d2gTrain(size: String, order: String, seed: Int): Unit =
  trainTranscriber(D2GSetup(D2SModelConfiguration.named(size)).copy(nodeOrder = NodeOrder.named(order), seed = seed))

@main
def d2gEval(size: String): Unit = scoreTranscriber(D2GSetup(D2SModelConfiguration.named(size)), size, Split.Validation)

@main
def d2gTest(size: String): Unit = scoreTranscriber(D2GSetup(D2SModelConfiguration.named(size)), size, Split.Test)

@main
def d2gPlot(size: String): Unit = plotTranscriber(D2GSetup(D2SModelConfiguration.named(size)))
