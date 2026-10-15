package d2seq.config

import d2seq.*
import dataset.Corpus
import dataset.Runs

/** Setup for a D2Seq experiment. */
case class D2SeqSetup(
    corpus: Corpus,

    // Model configuration
    numLayers: Int = 3,
    numHeads: Int = 4,
    embedding: Int = 128,
    order: SequenceOrder = SequenceOrder.Random,

    // Train configuration
    checkpointRoot: String,
    /** 138M drawings in batches of 512, the last 27.6M of them the cooldown — about 90 passes over
      * Vitruvion's training split.
      */
    checkpointEverySamples: Int = 40_000 * 128,
    numSamples: Int = 1_080_000 * 128,
    batchSize: Int = 512,
    learningRate: Float = 3e-4f,
    /** Adam's memory of how large the gradients are, about 1 / (1 − β₂) steps. Short enough that a
      * sudden growth of the gradients is caught up with within a few dozen steps, rather than
      * taking steps several times the learning rate for hundreds of them.
      */
    adamBeta2: Float = 0.98f,
    finalLearningRate: Float = 0f,
    /** AdamW's decoupled weight decay, taken per step times the learning rate. Strong enough to keep
      * the weights, and with them the logits and activations, from growing without bound.
      */
    weightDecay: Float = 0.05f,
    maxGradientNorm: Float = 1f,
    warmupSamples: Int = 1_000 * 128,
    cooldownSamples: Int = 216_000 * 128,
    seed: Int = 0
):

  val nodeSlots: Int = corpus.maxNodes + 1 // One more for saying the record ends

  override def toString: String = s"D2SeqSetup(${corpus.repoId}, layers=$numLayers, order=${order.name}, heads=$numHeads, embedding=$embedding, nodes=$nodeSlots, samples=$numSamples, batch=$batchSize, seed=$seed)"

object D2SeqSetup:

  def apply(corpus: Corpus, size: D2SeqModelConfiguration): D2SeqSetup = D2SeqSetup(
    corpus = corpus,
    checkpointRoot = s"${Runs.checkpointDir}/d2seq/${corpus.name}-${size.name}",
    numLayers = size.numLayers,
    numHeads = size.numHeads,
    embedding = size.embedding
  )

// Model sizes for experiments
enum D2SeqModelConfiguration(val name: String, val embedding: Int, val numLayers: Int, val numHeads: Int):
  case XS extends D2SeqModelConfiguration("xs", 128, 3, 4)
  case S extends D2SeqModelConfiguration("s", 256, 3, 8)
  case SDeep extends D2SeqModelConfiguration("s-deep", 256, 6, 8)
  case MDeep extends D2SeqModelConfiguration("m-deep", 512, 6, 8)

object D2SeqModelConfiguration:
  def named(name: String): D2SeqModelConfiguration = values.find(_.name == name).getOrElse(sys.error(s"no size named '$name': ${values.map(_.name).mkString(", ")}"))
