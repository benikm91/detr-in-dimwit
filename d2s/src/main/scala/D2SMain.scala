import d2s.*
import d2s.model.*
import d2s.train.*
import d2s.eval.*
import d2s.config.*
import dataset.Corpus

/** Entry points for the D2S experiments, on a corpus (e.g., `sketch-xl`) at a model configuration (e.g., `s-deep`): `sbt "d2s/runMain d2sTrain sketch-xl s-deep"`.
  * See [[Corpus]] and [[D2SModelConfiguration]] for the available configurations
  *
  * [[d2sTrain]] trains a run.
  * [[d2sEval]] scores the newest run of that corpus and size (greedy decoding).
  * [[d2sPlot]] shows what it transcribes (visualization).
  */

@main
def d2sTrain(corpus: String, size: String): Unit = trainSetTranscriber(D2SSetup(Corpus.named(corpus), D2SModelConfiguration.named(size)))

@main
def d2sEval(corpus: String, size: String): Unit = scoreSetTranscriber(D2SSetup(Corpus.named(corpus), D2SModelConfiguration.named(size)), size)

@main
def d2sPlot(corpus: String, size: String): Unit = plotSetTranscriber(D2SSetup(Corpus.named(corpus), D2SModelConfiguration.named(size)))
