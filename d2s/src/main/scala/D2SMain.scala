import d2s.*
import d2s.model.*
import d2s.train.*
import d2s.eval.*
import d2s.config.*
import dataset.Corpus
import dataset.DrawingDataset.Split

/** Entry points for the D2S experiments, on a corpus (e.g., `sketch-xl`) at a model configuration (e.g., `s-deep`): `sbt "d2s/runMain d2sTrain sketch-xl s-deep 0"`.
  * See [[Corpus]] and [[D2SModelConfiguration]] for the available configurations
  *
  * [[d2sTrain]] trains a run from a seed, so that a setting can be run again.
  * [[d2sEval]] scores the newest run of that corpus and size (greedy decoding).
  * [[d2sTest]] scores its last checkpoint on the test split.
  * [[d2sPlot]] shows what it transcribes (visualization).
  */

@main
def d2sTrain(corpus: String, size: String, seed: Int): Unit = trainSetTranscriber(D2SSetup(Corpus.named(corpus), D2SModelConfiguration.named(size)).copy(seed = seed))

@main
def d2sEval(corpus: String, size: String): Unit = scoreSetTranscriber(D2SSetup(Corpus.named(corpus), D2SModelConfiguration.named(size)), size, Split.Validation)

@main
def d2sTest(corpus: String, size: String): Unit = scoreSetTranscriber(D2SSetup(Corpus.named(corpus), D2SModelConfiguration.named(size)), size, Split.Test)

@main
def d2sPlot(corpus: String, size: String): Unit = plotSetTranscriber(D2SSetup(Corpus.named(corpus), D2SModelConfiguration.named(size)))
