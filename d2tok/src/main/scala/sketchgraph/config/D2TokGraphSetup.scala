package d2tok.sketchgraph.config

import d2tok.config.D2TokModelConfiguration
import d2tok.sketch.SequenceOrder
import d2tok.sketchgraph.*
import dataset.Corpus
import dataset.NodeClass
import dataset.NodeClasses
import dataset.Runs
import dimwit.*

/** Setup for a D2Tok experiment that writes a drawing's relationships after its nodes. */
case class D2TokGraphSetup(
    corpus: Corpus,

    // Model configuration
    numLayers: Int = 3,
    numHeads: Int = 4,
    embedding: Int = 128,
    /** Random or class: the orders that renumber the relationships along with the nodes. */
    order: SequenceOrder = SequenceOrder.ByClass,
    relationshipOrder: RelationshipOrder = RelationshipOrder.ByKind,

    // Train configuration: the schedule of d2s and d2g on the same corpora.
    checkpointRoot: String,
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
    seed: Int = 0
):

  require(order == SequenceOrder.Random || order == SequenceOrder.ByClass, s"the order ${order.name} does not renumber the relationships along with the nodes")

  /** The rank of every node class, by which `order` lays the nodes out. */
  def nodeClassRank: Tensor1[NodeClasses, Float32] =
    if order == SequenceOrder.ByClass then SequenceOrder.classRank else NodeClass.indicator(VType[Float32])(!_.isNode)

  val words: GraphWords = GraphWords(corpus.canvas, corpus.maxNodes)
  val tokens: Int = words.tokens(corpus.maxNodes, corpus.maxEdges)

  override def toString: String = s"D2TokGraphSetup(${corpus.repoId}, layers=$numLayers, order=${order.name}, relationship order=$relationshipOrder, heads=$numHeads, embedding=$embedding, words=${words.size}, tokens=$tokens, samples=$numSamples, batch=$batchSize, seed=$seed)"

object D2TokGraphSetup:

  def apply(corpus: Corpus, size: D2TokModelConfiguration): D2TokGraphSetup = D2TokGraphSetup(
    corpus = corpus,
    checkpointRoot = s"${Runs.checkpointDir}/d2tok-graph/${corpus.name}-${size.name}",
    numLayers = size.numLayers,
    numHeads = size.numHeads,
    embedding = size.embedding
  )
