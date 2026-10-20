package d2tok.floorplan.config

import d2tok.*
import d2tok.config.D2TokModelConfiguration
import d2tok.floorplan.*
import dataset.Runs
import syntheticFloorplansDataset.SyntheticFloorplans

/** Setup for a D2Tok experiment on the synthetic floor plans. */
case class D2TokSetup(
    // Model configuration
    numLayers: Int = 3,
    numHeads: Int = 4,
    embedding: Int = 128,
    /** The side the drawings are read at, in pixels: the stored ones averaged down. It is also how
      * many pixels a coordinate chooses from.
      */
    canvas: Int = 128,
    order: SequenceOrder = SequenceOrder.Random,

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

  val words: Words = Words(canvas)
  val tokens: Int = words.tokens(SyntheticFloorplans.maxNodes)

  override def toString: String = s"D2TokSetup(${SyntheticFloorplans.repoId}, layers=$numLayers, order=${order.name}, heads=$numHeads, embedding=$embedding, canvas=$canvas, words=${words.size}, tokens=$tokens, samples=$numSamples, batch=$batchSize, seed=$seed)"

object D2TokSetup:

  def apply(size: D2TokModelConfiguration): D2TokSetup = D2TokSetup(
    checkpointRoot = s"${Runs.checkpointDir}/d2tok/${SyntheticFloorplans.name}-${size.name}",
    numLayers = size.numLayers,
    numHeads = size.numHeads,
    embedding = size.embedding
  )
