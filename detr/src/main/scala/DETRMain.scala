import detr.*
import detr.model.*
import detr.train.*
import detr.eval.*
import detr.config.*
import dataset.Corpus
import dataset.DrawingDataset.Split

/** The detector, on a corpus at a size — `sbt "detr/runMain detrTrain sketch s"`.
  *
  * [[detrTrain]] trains a run, [[detrEval]] scores the newest run of that corpus and size,
  * [[detrTest]] scores its last checkpoint on the test split, and [[detrPlot]] shows what it
  * detects. See [[Corpus]] and [[DETR.Size]] for the names.
  */
@main
def detrTrain(corpus: String, size: String): Unit = trainDetector(DETRSetup(Corpus.named(corpus), DETR.Size.named(size)))

@main
def detrEval(corpus: String, size: String): Unit = scoreDetector(DETRSetup(Corpus.named(corpus), DETR.Size.named(size)), size, Split.Validation)

@main
def detrTest(corpus: String, size: String): Unit = scoreDetector(DETRSetup(Corpus.named(corpus), DETR.Size.named(size)), size, Split.Test)

@main
def detrPlot(corpus: String, size: String): Unit = plotDetector(DETRSetup(Corpus.named(corpus), DETR.Size.named(size)))
