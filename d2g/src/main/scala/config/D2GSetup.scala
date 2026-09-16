package d2g.config

import d2g.*
import d2g.model.*
import d2g.train.*
import d2g.eval.*
import dataset.Corpus
import dataset.Runs

/** Setup for a D2G experiment. */
case class D2GSetup(
    corpus: Corpus,

    // Model configuration
    numLayers: Int = 3,
    numHeads: Int = 4,
    embedding: Int = 128,
    patchSize: Int = 16,
    queryPool: Int = 6,

    // Train configuration
    checkpointRoot: String,
    checkpointEvery: Int = 10_000,
    numIterations: Int = 200_000,
    batchSize: Int = 64,
    learningRate: Float = 3e-4f,
    finalLearningRate: Float = 1e-4f,
    weightDecay: Float = 1e-4f,
    maxGradientNorm: Float = 1f,
    warmupSteps: Int = 2_000,
    seed: Int = 42
):

  val nodeSlots: Int = corpus.maxNodes + 1 // One more due to the end prediction
  val edgeSlots: Int = corpus.maxEdges + 1 // One more due to the end prediction

  override def toString: String = s"D2GSetup(${corpus.repoId}, layers=$numLayers, heads=$numHeads, embedding=$embedding, nodes=$nodeSlots, edges=$edgeSlots, iterations=$numIterations, batch=$batchSize)"

object D2GSetup:

  def apply(corpus: Corpus, size: D2GModelConfiguration): D2GSetup = D2GSetup(
    corpus = corpus,
    checkpointRoot = s"${Runs.checkpointDir}/d2g/${corpus.name}-${size.name}",
    numLayers = size.numLayers,
    numHeads = size.numHeads,
    embedding = size.embedding
  )

// Model sizes for experiments
enum D2GModelConfiguration(val name: String, val embedding: Int, val numLayers: Int, val numHeads: Int):
  case XS extends D2GModelConfiguration("xs", 128, 3, 4)
  case S extends D2GModelConfiguration("s", 256, 3, 8)

object D2GModelConfiguration:
  def named(name: String): D2GModelConfiguration = values.find(_.name == name).getOrElse(sys.error(s"no size named '$name': ${values.map(_.name).mkString(", ")}"))
