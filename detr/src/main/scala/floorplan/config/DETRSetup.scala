package detr.floorplan.config

import detr.config.DETRModelConfiguration
import dataset.Runs
import syntheticFloorplansDataset.SyntheticFloorplans

/** Everything a training run of the detector on the synthetic floor plans needs to know beyond the
  * model itself.
  */
case class DETRSetup(
    /** Where a run of this setup writes its checkpoints. One root per size, so that a run is never
      * read back into a model of another shape.
      */
    checkpointRoot: String,

    /** How many objects the decoder may answer with: one query per node a plan can hold. */
    numQueries: Int = SyntheticFloorplans.maxNodes,
    numLayers: Int = 3,
    numHeads: Int = 4,
    embedding: Int = 128,
    /** The side the drawings are read at, in pixels: the stored ones averaged down. It is also how
      * many pixels a coordinate chooses from.
      */
    canvas: Int = 128,

    /** 100k steps of 512 plans, about 114 passes over the training split: 10k steps of warmup, 70k
      * at the full learning rate and a 20k cooldown, with a checkpoint every 10k.
      */
    checkpointEverySamples: Int = 40_000 * 128,
    numSamples: Int = 400_000 * 128,
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

    /** Global L2 norm the gradients are rescaled to, as in the DETR paper. The set loss reassigns
      * which query is responsible for which object from step to step, so a batch that reshuffles
      * the matching produces a far larger gradient than a batch that confirms it; clipping keeps
      * those steps from undoing what the settled ones learned.
      */
    maxGradientNorm: Float = 1f,
    warmupSamples: Int = 40_000 * 128,
    cooldownSamples: Int = 80_000 * 128,

    seed: Int = 0
):
  require(numQueries >= SyntheticFloorplans.maxNodes, s"$numQueries queries cannot answer for a plan of up to ${SyntheticFloorplans.maxNodes} nodes")

  override def toString: String =
    s"DETRSetup(${SyntheticFloorplans.repoId}, layers=$numLayers, heads=$numHeads, embedding=$embedding, canvas=$canvas, " +
      s"queries=$numQueries, samples=$numSamples, batch=$batchSize)"

object DETRSetup:

  def apply(size: DETRModelConfiguration): DETRSetup = DETRSetup(
    checkpointRoot = s"${Runs.checkpointDir}/detr/${SyntheticFloorplans.name}-${size.name}",
    numLayers = size.numLayers,
    numHeads = size.numHeads,
    embedding = size.embedding
  )
