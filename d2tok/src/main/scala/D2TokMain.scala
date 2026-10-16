import d2tok.config.D2TokModelConfiguration
import dataset.Corpus
import dataset.DrawingDataset.Split
import syntheticFloorplansDataset.SyntheticFloorplans

/** Entry points for the D2Tok experiments, on a corpus (e.g., `vitruvion-primitives` or
  * `synthetic-floorplans`) at a model configuration (e.g., `s-deep`):
  * `sbt "d2tok/runMain d2tokTrain vitruvion-primitives s-deep random 0"`.
  *
  * Each kind of record has a language of its own: the sketch corpora (see [[Corpus]]) are written by
  * [[d2tok.sketch]], the synthetic floor plans by [[d2tok.floorplan]]. The corpus names which, and
  * the order is one of that language's: [[d2tok.sketch.SequenceOrder]] or
  * [[d2tok.floorplan.SequenceOrder]]. See [[D2TokModelConfiguration]] for the sizes.
  *
  * [[d2tokTrain]] trains a run in an order, from a seed.
  * [[d2tokEval]] scores the newest run of that corpus and size (greedy decoding).
  * [[d2tokTest]] scores its last checkpoint on the test split.
  */

@main
def d2tokTrain(corpus: String, size: String, order: String, seed: Int): Unit =
  if corpus == SyntheticFloorplans.name then
    val setup = d2tok.floorplan.config.D2TokSetup(D2TokModelConfiguration.named(size))
    d2tok.floorplan.train.trainWordTranscriber(setup.copy(order = d2tok.floorplan.SequenceOrder.named(order), seed = seed))
  else
    val setup = d2tok.sketch.config.D2TokSetup(Corpus.named(corpus), D2TokModelConfiguration.named(size))
    d2tok.sketch.train.trainWordTranscriber(setup.copy(order = d2tok.sketch.SequenceOrder.named(order), seed = seed))

@main
def d2tokEval(corpus: String, size: String): Unit = score(corpus, size, Split.Validation)

@main
def d2tokTest(corpus: String, size: String): Unit = score(corpus, size, Split.Test)

private def score(corpus: String, size: String, split: Split): Unit =
  if corpus == SyntheticFloorplans.name then
    d2tok.floorplan.eval.scoreWordTranscriber(d2tok.floorplan.config.D2TokSetup(D2TokModelConfiguration.named(size)), size, split)
  else
    d2tok.sketch.eval.scoreWordTranscriber(d2tok.sketch.config.D2TokSetup(Corpus.named(corpus), D2TokModelConfiguration.named(size)), size, split)
