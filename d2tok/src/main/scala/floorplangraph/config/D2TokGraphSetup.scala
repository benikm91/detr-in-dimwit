package d2tok.floorplangraph.config

import d2tok.config.D2TokModelConfiguration
import d2tok.floorplan.SequenceOrder
import d2tok.floorplangraph.*
import dataset.Runs
import dimwit.*
import syntheticFloorplansDataset.FloorplanClass
import syntheticFloorplansDataset.FloorplanClasses
import syntheticFloorplansDataset.FloorplanCorpus

/** Setup for a D2Tok experiment that writes a floor plan's relationships after its nodes. */
case class D2TokGraphSetup(
    corpus: FloorplanCorpus,

    // Model configuration
    numLayers: Int = 3,
    numHeads: Int = 4,
    embedding: Int = 128,
    /** The side the drawings are read at, in pixels: the stored ones averaged down. It is also how
      * many pixels a coordinate chooses from.
      */
    canvas: Int = 128,
    /** Random or walls-first: the orders that renumber the relationships along with the nodes. */
    nodeOrder: SequenceOrder = SequenceOrder.WallsFirst,
    relationshipOrder: RelationshipOrder = RelationshipOrder.ByKind,

    // Train configuration: the schedule of every model on the floor plans.
    checkpointRoot: String,
    checkpointEverySamples: Int = 40_000 * 128,
    numSamples: Int = 800_000 * 128,
    batchSize: Int = 512,
    learningRate: Float = 3e-4f,
    adamBeta2: Float = 0.98f,
    finalLearningRate: Float = 0f,
    weightDecay: Float = 0.05f,
    maxGradientNorm: Float = 1f,
    warmupSamples: Int = 40_000 * 128,
    cooldownSamples: Int = 160_000 * 128,
    seed: Int = 0
):

  require(nodeOrder == SequenceOrder.Random || nodeOrder == SequenceOrder.WallsFirst, s"the order ${nodeOrder.name} does not renumber the relationships along with the nodes")

  /** The rank of every node class, by which `nodeOrder` lays the nodes out. */
  def nodeClassRank: Tensor1[FloorplanClasses, Float32] =
    if nodeOrder == SequenceOrder.WallsFirst then SequenceOrder.classRank else FloorplanClass.indicator(VType[Float32])(!_.isNode)

  val words: GraphWords = GraphWords(canvas, corpus.maxNodes)
  val tokens: Int = words.tokens(corpus.maxNodes, corpus.maxRelationships)

  override def toString: String = s"D2TokGraphSetup(${corpus.repoId}, layers=$numLayers, nodeOrder=${nodeOrder.name}, relationshipOrder=$relationshipOrder, heads=$numHeads, embedding=$embedding, canvas=$canvas, words=${words.size}, tokens=$tokens, samples=$numSamples, batch=$batchSize, seed=$seed)"

object D2TokGraphSetup:

  def apply(corpus: FloorplanCorpus, size: D2TokModelConfiguration): D2TokGraphSetup = D2TokGraphSetup(
    corpus = corpus,
    checkpointRoot = s"${Runs.checkpointDir}/d2tok-graph/${corpus.name}-${size.name}",
    numLayers = size.numLayers,
    numHeads = size.numHeads,
    embedding = size.embedding
  )
