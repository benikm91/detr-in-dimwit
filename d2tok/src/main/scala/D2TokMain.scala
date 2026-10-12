import d2tok.*
import d2tok.train.*
import d2tok.eval.*
import d2tok.config.*
import dataset.Corpus
import dataset.DrawingDataset.Split

/** Entry points for the D2Tok experiments, on a corpus (e.g., `vitruvion-primitives`) at a model
  * configuration (e.g., `s-deep`): `sbt "d2tok/runMain d2tokTrain vitruvion-primitives s-deep random 0"`.
  * See [[Corpus]], [[D2TokModelConfiguration]] and [[SequenceOrder]] for the available names.
  *
  * [[d2tokTrain]] trains a run in an order, from a seed.
  * [[d2tokEval]] scores the newest run of that corpus and size (greedy decoding).
  * [[d2tokTest]] scores its last checkpoint on the test split.
  */

@main
def d2tokTrain(corpus: String, size: String, order: String, seed: Int): Unit =
  trainWordTranscriber(D2TokSetup(Corpus.named(corpus), D2TokModelConfiguration.named(size)).copy(order = SequenceOrder.named(order), seed = seed))

@main
def d2tokEval(corpus: String, size: String): Unit = scoreWordTranscriber(D2TokSetup(Corpus.named(corpus), D2TokModelConfiguration.named(size)), size, Split.Validation)

@main
def d2tokTest(corpus: String, size: String): Unit = scoreWordTranscriber(D2TokSetup(Corpus.named(corpus), D2TokModelConfiguration.named(size)), size, Split.Test)
