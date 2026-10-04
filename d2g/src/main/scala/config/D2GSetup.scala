package d2g.config

import d2g.*
import d2g.model.*
import d2g.train.*
import d2g.eval.*
import d2s.config.D2SModelConfiguration
import dataset.Corpus
import dataset.Runs

/** Setup for a D2G experiment. */
case class D2GSetup(
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
    learningRate: Float = 3e-4f,
    finalLearningRate: Float = 0f,
    weightDecay: Float = 1e-4f,
    maxGradientNorm: Float = 1f,
    warmupSamples: Int = 1_000 * 128,
    cooldownSamples: Int = 100_000 * 128,
    seed: Int = 42
):

  val nodeSlots: Int = corpus.maxNodes + 1 // One more due to the end prediction
  val edgeSlots: Int = corpus.maxEdges + 1 // One more due to the end prediction

  override def toString: String = s"D2GSetup(${corpus.repoId}, layers=$numLayers, heads=$numHeads, embedding=$embedding, nodes=$nodeSlots, edges=$edgeSlots, samples=$numSamples, batch=$batchSize)"

object D2GSetup:

  def apply(corpus: Corpus, size: D2SModelConfiguration): D2GSetup = D2GSetup(
    corpus = corpus,
    checkpointRoot = s"${Runs.checkpointDir}/d2g/${corpus.name}-${size.name}",
    numLayers = size.numLayers,
    numHeads = size.numHeads,
    embedding = size.embedding
  )
