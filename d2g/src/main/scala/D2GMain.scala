import d2g.*
import d2g.model.*
import d2g.train.*
import d2g.eval.*
import d2g.config.*
import dataset.Corpus

/** Entry points for the D2G experiments, on a corpus (e.g., `lshape`) at a model configuration (e.g., `s`): `sbt "d2g/runMain d2gTrain lshape s"`.
  * See [[Corpus]] and [[D2GModelConfiguration]] for the available configurations
  *
  * [[d2gTrain]] trains a run.
  * [[d2gContinue]] takes the newest run up again at one of its checkpoints and trains it longer.
  * [[d2gEval]] scores the newest run of that corpus and size (greedy decoding).
  * [[d2gPlot]] shows what it transcribes (visualization).
  */

@main
def d2gTrain(corpus: String, size: String): Unit = trainTranscriber(D2GSetup(Corpus.named(corpus), D2GModelConfiguration.named(size)))

/** From a checkpoint at the constant learning rate: `moreSteps` at it, then a cooldown over
  * `cooldownSteps`.
  */
@main
def d2gContinue(corpus: String, size: String, step: Int): Unit =
  val setup = D2GSetup(Corpus.named(corpus), D2GModelConfiguration.named(size))
  val moreSteps = 120_000
  val cooldownSteps = 80_000
  val longer = setup.copy(numSamples = (step + moreSteps + cooldownSteps) * setup.batchSize, cooldownSamples = cooldownSteps * setup.batchSize)
  trainTranscriber(longer, from = Some(step))

@main
def d2gEval(corpus: String, size: String): Unit = scoreTranscriber(D2GSetup(Corpus.named(corpus), D2GModelConfiguration.named(size)), size)

@main
def d2gPlot(corpus: String, size: String): Unit = plotTranscriber(D2GSetup(Corpus.named(corpus), D2GModelConfiguration.named(size)))
