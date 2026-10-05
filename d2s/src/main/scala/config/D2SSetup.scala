package d2s.config

import d2s.*
import d2s.model.*
import dataset.Corpus
import dataset.Runs

/** Setup for a D2S experiment. */
case class D2SSetup(
    corpus: Corpus,

    // Model configuration
    numLayers: Int = 3,
    numHeads: Int = 4,
    embedding: Int = 128,
    queryPool: Int = 6,

    // Train configuration
    checkpointRoot: String,
    checkpointEverySamples: Int = 20_000 * 128,
    numSamples: Int = 500_000 * 128,
    batchSize: Int = 512,
    learningRate: Float = 1e-4f,
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
    cooldownSamples: Int = 100_000 * 128,
    seed: Int = 42
):

  val nodeSlots: Int = corpus.maxNodes + 1 // One more due to the end prediction

  override def toString: String = s"D2SSetup(${corpus.repoId}, layers=$numLayers, heads=$numHeads, embedding=$embedding, nodes=$nodeSlots, samples=$numSamples, batch=$batchSize)"

object D2SSetup:

  def apply(corpus: Corpus, size: D2SModelConfiguration): D2SSetup = D2SSetup(
    corpus = corpus,
    checkpointRoot = s"${Runs.checkpointDir}/d2s/${corpus.name}-${size.name}",
    numLayers = size.numLayers,
    numHeads = size.numHeads,
    embedding = size.embedding
  )

// Model sizes for experiments
enum D2SModelConfiguration(val name: String, val embedding: Int, val numLayers: Int, val numHeads: Int):
  case XS extends D2SModelConfiguration("xs", 128, 3, 4)
  case S extends D2SModelConfiguration("s", 256, 3, 8)
  case SDeep extends D2SModelConfiguration("s-deep", 256, 6, 8)
  case MDeep extends D2SModelConfiguration("m-deep", 512, 6, 8)

object D2SModelConfiguration:
  def named(name: String): D2SModelConfiguration = values.find(_.name == name).getOrElse(sys.error(s"no size named '$name': ${values.map(_.name).mkString(", ")}"))
