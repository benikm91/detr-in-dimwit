import d2seq.*
import d2seq.train.*
import d2seq.eval.*
import d2seq.config.*
import dataset.Corpus
import dataset.DrawingDataset.Split

/** Entry points for the D2Seq experiments, on a corpus (e.g., `vitruvion-primitives`) at a model
  * configuration (e.g., `s-deep`): `sbt "d2seq/runMain d2seqTrain vitruvion-primitives s-deep random 0"`.
  * See [[Corpus]], [[D2SeqModelConfiguration]] and [[SequenceOrder]] for the available names.
  *
  * [[d2seqTrain]] trains a run in an order, from a seed.
  * [[d2seqEval]] scores the newest run of that corpus and size (greedy decoding).
  * [[d2seqTest]] scores its last checkpoint on the test split.
  */

@main
def d2seqTrain(corpus: String, size: String, order: String, seed: Int): Unit =
  trainSequenceTranscriber(D2SeqSetup(Corpus.named(corpus), D2SeqModelConfiguration.named(size)).copy(order = SequenceOrder.named(order), seed = seed))

@main
def d2seqEval(corpus: String, size: String): Unit = scoreSequenceTranscriber(D2SeqSetup(Corpus.named(corpus), D2SeqModelConfiguration.named(size)), size, Split.Validation)

@main
def d2seqTest(corpus: String, size: String): Unit = scoreSequenceTranscriber(D2SeqSetup(Corpus.named(corpus), D2SeqModelConfiguration.named(size)), size, Split.Test)
