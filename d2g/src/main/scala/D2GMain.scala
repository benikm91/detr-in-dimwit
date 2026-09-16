import d2g.*
import d2g.model.*
import d2g.train.*
import d2g.eval.*
import d2g.config.*
import dataset.Corpus

/** The transcription model, on a corpus at a size — `sbt "d2g/runMain d2gTrain sketch s"`.
  *
  * [[d2gTrain]] trains a run, [[d2gEval]] scores the newest run of that corpus and size, taking
  * the best answer the query pool offers at every slot, and [[d2gPlot]] shows what it transcribes.
  * See [[Corpus]] and [[D2G.Size]] for the names.
  */
@main
def d2gTrain(corpus: String, size: String): Unit = trainTranscriber(D2GSetup(Corpus.named(corpus), D2G.Size.named(size)))

@main
def d2gEval(corpus: String, size: String): Unit = scoreTranscriber(D2GSetup(Corpus.named(corpus), D2G.Size.named(size)), size, width = 1)

@main
def d2gPlot(corpus: String, size: String): Unit = plotTranscriber(D2GSetup(Corpus.named(corpus), D2G.Size.named(size)))
