import d2g.floorplan.config.D2GSetup
import d2s.config.D2SModelConfiguration
import d2s.floorplan.NodeOrder
import dataset.DrawingDataset.Split

@main def d2gSmoke(): Unit =
  val setup = D2GSetup(D2SModelConfiguration.named("xs")).copy(nodeOrder = NodeOrder.WallsFirst, batchSize = 8, numSamples = 16 * 8, checkpointEverySamples = 8 * 8, warmupSamples = 2 * 8, cooldownSamples = 4 * 8)
  d2g.floorplan.train.trainTranscriber(setup)
  d2g.floorplan.eval.scoreTranscriber(setup, "xs", Split.Validation)
