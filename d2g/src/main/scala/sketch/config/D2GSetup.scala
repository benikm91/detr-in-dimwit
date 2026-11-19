package d2g.sketch.config

import d2g.sketch.RelationshipOrder
import d2s.config.D2SModelConfiguration
import d2s.sketch.NodeOrder
import dataset.Corpus
import dataset.Runs

/** Setup for a D2G experiment on a corpus of drawings whose records hold relationships. */
case class D2GSetup(
    corpus: Corpus,

    // Model configuration
    numLayers: Int = 3,
    numHeads: Int = 4,
    embedding: Int = 128,
    queryPool: Int = 6,
    /** The order the d2s run whose nodes the relationships are written on wrote them in. */
    nodeOrder: NodeOrder = NodeOrder.SimplestFirst,
    /** Whether the relationships of a record are written down in any order or class by class. */
    relationshipOrder: RelationshipOrder = RelationshipOrder.Free,

    // Train configuration: the schedule of the d2s run below it.
    checkpointRoot: String,
    /** The run of a [[d2s.sketch.model.D2S]] whose newest checkpoint writes the nodes. Its weights
      * stay as they are: only the relationships are trained.
      */
    nodesFrom: String,
    checkpointEverySamples: Int = 20_000 * 128,
    numSamples: Int = 500_000 * 128,
    batchSize: Int = 512,
    learningRate: Float = 3e-4f,
    adamBeta2: Float = 0.98f,
    finalLearningRate: Float = 0f,
    weightDecay: Float = 0.05f,
    maxGradientNorm: Float = 1f,
    warmupSamples: Int = 1_000 * 128,
    cooldownSamples: Int = 100_000 * 128,
    seed: Int = 42
):

  val nodeSlots: Int = corpus.maxNodes + 1 // One more due to the end prediction
  val edgeSlots: Int = corpus.maxEdges + 1 // One more due to the end prediction

  override def toString: String = s"D2GSetup(${corpus.repoId}, layers=$numLayers, order=$nodeOrder, relationship order=$relationshipOrder, heads=$numHeads, embedding=$embedding, nodes=$nodeSlots, edges=$edgeSlots, samples=$numSamples, batch=$batchSize)"

object D2GSetup:

  def apply(corpus: Corpus, size: D2SModelConfiguration): D2GSetup = D2GSetup(
    corpus = corpus,
    checkpointRoot = s"${Runs.checkpointDir}/d2g/${corpus.name}-${size.name}",
    nodesFrom = s"${sys.env.getOrElse("NODES_DIR", Runs.checkpointDir)}/d2s/${corpus.name}-${size.name}",
    numLayers = size.numLayers,
    numHeads = size.numHeads,
    embedding = size.embedding
  )
