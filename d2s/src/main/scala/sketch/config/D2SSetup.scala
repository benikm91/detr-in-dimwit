package d2s.sketch.config

import d2s.*
import d2s.sketch.*
import d2s.model.*
import d2s.sketch.model.*
import d2s.config.D2SModelConfiguration
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
    /** Whether the nodes of a record are written down in any order or in an order of their classes. */
    nodeOrder: NodeOrder = NodeOrder.SimplestFirst,

    // Train configuration
    checkpointRoot: String,
    checkpointEverySamples: Int = 20_000 * 128,
    numSamples: Int = 500_000 * 128,
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
    cooldownSamples: Int = 100_000 * 128,
    seed: Int = 42
):

  val nodeSlots: Int = corpus.maxNodes + 1 // One more due to the end prediction

  override def toString: String = s"D2SSetup(${corpus.repoId}, layers=$numLayers, order=$nodeOrder, heads=$numHeads, embedding=$embedding, nodes=$nodeSlots, samples=$numSamples, batch=$batchSize)"

object D2SSetup:

  def apply(corpus: Corpus, size: D2SModelConfiguration): D2SSetup = D2SSetup(
    corpus = corpus,
    checkpointRoot = s"${Runs.checkpointDir}/d2s/${corpus.name}-${size.name}",
    numLayers = size.numLayers,
    numHeads = size.numHeads,
    embedding = size.embedding
  )
