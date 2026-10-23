package d2tok.floorplangraph.config

import d2tok.*
import d2tok.config.D2TokModelConfiguration
import d2tok.floorplan.SequenceOrder
import d2tok.floorplan.Words
import d2tok.floorplangraph.*
import dataset.Runs
import syntheticFloorplansDataset.SyntheticFloorplans

/** Setup for a D2Tok experiment on the synthetic floor plans that writes their relationships too. */
case class D2TokGraphSetup(
    // Model configuration
    numLayers: Int = 3,
    numHeads: Int = 4,
    embedding: Int = 128,
    /** The side the drawings are read at, in pixels: the stored ones averaged down. It is also how
      * many pixels a coordinate chooses from.
      */
    canvas: Int = 128,
    nodeOrder: SequenceOrder = SequenceOrder.Random,
    relationshipOrder: RelationshipOrder = RelationshipOrder.Random,

    // Train configuration
    checkpointRoot: String,
    /** 200k steps of 512 plans, about 228 passes over the training split: 10k steps of warmup, 150k
      * at the full learning rate and a 40k cooldown, with a checkpoint every 10k.
      */
    checkpointEverySamples: Int = 40_000 * 128,
    numSamples: Int = 800_000 * 128,
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
    warmupSamples: Int = 40_000 * 128,
    cooldownSamples: Int = 160_000 * 128,
    seed: Int = 0
):

  // The position order moves the nodes without renumbering the relationships that name them.
  require(nodeOrder != SequenceOrder.ByPosition, s"the relationships cannot follow the nodes in the order '${nodeOrder.name}'")

  val words: Words = Words(canvas)
  val tokens: Int = words.tokens(SyntheticFloorplans.maxNodes)

  val relationshipWords: RelationshipWords = RelationshipWords(SyntheticFloorplans.maxNodes)
  val relationshipTokens: Int = relationshipWords.tokens(SyntheticFloorplans.maxRelationships)

  override def toString: String = s"D2TokGraphSetup(${SyntheticFloorplans.repoId}, layers=$numLayers, nodeOrder=${nodeOrder.name}, relationshipOrder=${relationshipOrder.name}, heads=$numHeads, embedding=$embedding, canvas=$canvas, words=${words.size}+${relationshipWords.size}, tokens=$tokens+$relationshipTokens, samples=$numSamples, batch=$batchSize, seed=$seed)"

object D2TokGraphSetup:

  /** What a run names the corpus by on its command line and in what it writes. */
  val corpus = s"${SyntheticFloorplans.name}-graph"

  def apply(size: D2TokModelConfiguration): D2TokGraphSetup = D2TokGraphSetup(
    checkpointRoot = s"${Runs.checkpointDir}/d2tok/$corpus-${size.name}",
    numLayers = size.numLayers,
    numHeads = size.numHeads,
    embedding = size.embedding
  )
