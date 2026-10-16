package d2g.floorplan.config

import d2s.config.D2SModelConfiguration
import d2s.floorplan.NodeOrder
import dataset.Runs
import syntheticFloorplansDataset.SyntheticFloorplans

/** Setup for a D2G experiment on the synthetic floor plans. */
case class D2GSetup(
    // Model configuration
    numLayers: Int = 3,
    numHeads: Int = 4,
    embedding: Int = 128,
    queryPool: Int = 6,
    /** The side the drawings are read at, in pixels: the stored ones averaged down. It is also how
      * many pixels a coordinate chooses from.
      */
    canvas: Int = 128,
    /** Whether the nodes of a record are written down in any order or in an order of their classes. */
    nodeOrder: NodeOrder = NodeOrder.WallsFirst,

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

  val nodeSlots: Int = SyntheticFloorplans.maxNodes + 1 // One more due to the end prediction
  val edgeSlots: Int = SyntheticFloorplans.maxRelationships + 1 // One more due to the end prediction

  override def toString: String = s"D2GSetup(${SyntheticFloorplans.repoId}, layers=$numLayers, order=$nodeOrder, heads=$numHeads, embedding=$embedding, canvas=$canvas, nodes=$nodeSlots, edges=$edgeSlots, samples=$numSamples, batch=$batchSize)"

object D2GSetup:

  def apply(size: D2SModelConfiguration): D2GSetup = D2GSetup(
    checkpointRoot = s"${Runs.checkpointDir}/d2g/${SyntheticFloorplans.name}-${size.name}",
    numLayers = size.numLayers,
    numHeads = size.numHeads,
    embedding = size.embedding
  )
