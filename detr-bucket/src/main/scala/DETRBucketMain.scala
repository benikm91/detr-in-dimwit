import detr.config.DETRModelConfiguration
import detrbucket.sketch.config.DETRBucketSetup
import detrbucket.sketch.eval.*
import detrbucket.sketch.train.*
import dataset.Corpus
import dataset.DrawingDataset.Split

/** The detector with its queries split by class, on a sketch corpus at a size —
  * `sbt "detrBucket/runMain detrBucketTrain vitruvion-primitives s-deep 0"`.
  *
  * [[detrBucketTrain]] trains a run, [[detrBucketEval]] scores the newest run of that corpus and
  * size, [[detrBucketTest]] scores its last checkpoint on the test split, [[detrBucketPlot]] shows
  * what it detects, and [[detrBucketDraw]] writes what every checkpoint detected as a picture. See
  * [[Corpus]] and [[DETRModelConfiguration]] for the names.
  */
@main
def detrBucketTrain(corpus: String, size: String, seed: Int): Unit =
  trainBucketDetector(DETRBucketSetup(Corpus.named(corpus), DETRModelConfiguration.named(size)).copy(seed = seed))

@main
def detrBucketEval(corpus: String, size: String): Unit =
  scoreBucketDetector(DETRBucketSetup(Corpus.named(corpus), DETRModelConfiguration.named(size)), size, Split.Validation)

@main
def detrBucketTest(corpus: String, size: String): Unit =
  scoreBucketDetector(DETRBucketSetup(Corpus.named(corpus), DETRModelConfiguration.named(size)), size, Split.Test)

@main
def detrBucketPlot(corpus: String, size: String): Unit = plotBucketDetector(DETRBucketSetup(Corpus.named(corpus), DETRModelConfiguration.named(size)))

@main
def detrBucketDraw(corpus: String, size: String): Unit = drawBucketDetector(DETRBucketSetup(Corpus.named(corpus), DETRModelConfiguration.named(size)), size)
