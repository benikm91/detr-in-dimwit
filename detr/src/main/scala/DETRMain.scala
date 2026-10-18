import detr.config.DETRModelConfiguration
import dataset.Corpus
import dataset.DrawingDataset.Split
import syntheticFloorplansDataset.SyntheticFloorplans

/** The detector, on a corpus at a size — `sbt "detr/runMain detrTrain sketch s 0"`.
  *
  * Each kind of record has a detector of its own: the sketch corpora (see [[Corpus]]) are read by
  * [[detr.sketch]], the synthetic floor plans by [[detr.floorplan]]. See [[DETRModelConfiguration]]
  * for the sizes.
  *
  * [[detrTrain]] trains a run, [[detrEval]] scores the newest run of that corpus and size,
  * [[detrTest]] scores its last checkpoint on the test split, [[detrPlot]] shows what it detects,
  * and [[detrDraw]] writes what every checkpoint detected as a picture.
  */
@main
def detrTrain(corpus: String, size: String, seed: Int): Unit =
  if corpus == SyntheticFloorplans.name then
    detr.floorplan.train.trainDetector(detr.floorplan.config.DETRSetup(DETRModelConfiguration.named(size)).copy(seed = seed))
  else
    detr.sketch.train.trainDetector(detr.sketch.config.DETRSetup(Corpus.named(corpus), DETRModelConfiguration.named(size)).copy(seed = seed))

@main
def detrEval(corpus: String, size: String): Unit = score(corpus, size, Split.Validation)

@main
def detrTest(corpus: String, size: String): Unit = score(corpus, size, Split.Test)

@main
def detrPlot(corpus: String, size: String): Unit =
  if corpus == SyntheticFloorplans.name then detr.floorplan.eval.plotDetector(detr.floorplan.config.DETRSetup(DETRModelConfiguration.named(size)))
  else detr.sketch.eval.plotDetector(detr.sketch.config.DETRSetup(Corpus.named(corpus), DETRModelConfiguration.named(size)))

@main
def detrDraw(corpus: String, size: String): Unit =
  if corpus == SyntheticFloorplans.name then detr.floorplan.eval.drawDetector(detr.floorplan.config.DETRSetup(DETRModelConfiguration.named(size)), size)
  else detr.sketch.eval.drawDetector(detr.sketch.config.DETRSetup(Corpus.named(corpus), DETRModelConfiguration.named(size)), size)

private def score(corpus: String, size: String, split: Split): Unit =
  if corpus == SyntheticFloorplans.name then
    detr.floorplan.eval.scoreDetector(detr.floorplan.config.DETRSetup(DETRModelConfiguration.named(size)), size, split)
  else
    detr.sketch.eval.scoreDetector(detr.sketch.config.DETRSetup(Corpus.named(corpus), DETRModelConfiguration.named(size)), size, split)
