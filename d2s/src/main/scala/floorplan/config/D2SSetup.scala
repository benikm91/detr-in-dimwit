package d2s.floorplan.config

import d2s.config.D2SModelConfiguration
import d2s.floorplan.NodeOrder
import dataset.Runs
import syntheticFloorplansDataset.SyntheticFloorplans

/** Setup for a D2S experiment on the synthetic floor plans. */
case class D2SSetup(
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
    nodeOrder: NodeOrder = NodeOrder.Free,

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
    seed: Int = 42
):

  val nodeSlots: Int = SyntheticFloorplans.maxNodes + 1 // One more due to the end prediction

  override def toString: String = s"D2SSetup(${SyntheticFloorplans.repoId}, layers=$numLayers, order=$nodeOrder, heads=$numHeads, embedding=$embedding, canvas=$canvas, nodes=$nodeSlots, samples=$numSamples, batch=$batchSize)"

object D2SSetup:

  def apply(size: D2SModelConfiguration): D2SSetup = D2SSetup(
    checkpointRoot = s"${Runs.checkpointDir}/d2s/${SyntheticFloorplans.name}-${size.name}",
    numLayers = size.numLayers,
    numHeads = size.numHeads,
    embedding = size.embedding
  )
