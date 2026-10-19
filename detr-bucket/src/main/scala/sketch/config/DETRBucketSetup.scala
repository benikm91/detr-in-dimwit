package detrbucket.sketch.config

import detr.config.DETRModelConfiguration
import detrbucket.sketch.Buckets
import dataset.Corpus
import dataset.NodeClass
import dataset.Runs

/** Everything a training run of the bucketed detector needs to know beyond the model itself, with
  * the recipe the Vitruvion detector was trained with, so that the two are compared on it.
  */
case class DETRBucketSetup(
    corpus: Corpus,

    /** Where a run of this setup writes its checkpoints. One root per corpus and size, so that a
      * run is never read back into a model of another shape.
      */
    checkpointRoot: String,

    /** The queries of every class. A bucket holds as many as a drawing can hold nodes, so that
      * a drawing of nothing but that class can still be answered in full.
      */
    buckets: Buckets,
    numLayers: Int = 3,
    numHeads: Int = 4,
    embedding: Int = 128,

    /** 138M drawings in batches of 512, the last 27.6M of them the cooldown, with a checkpoint every
      * 10k steps.
      */
    checkpointEverySamples: Int = 40_000 * 128,
    numSamples: Int = 1_080_000 * 128,
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
    warmupSamples: Int = 1_000 * 128,
    cooldownSamples: Int = 216_000 * 128,

    seed: Int = 0
):
  require(
    buckets.sizes.forall(_._2 >= corpus.maxNodes),
    s"buckets of $buckets cannot answer for a drawing of up to ${corpus.maxNodes} nodes of one class"
  )

  override def toString: String =
    s"DETRBucketSetup(${corpus.repoId}, layers=$numLayers, heads=$numHeads, embedding=$embedding, " +
      s"queries=$buckets, objects<=${corpus.maxNodes}, samples=$numSamples, batch=$batchSize)"

object DETRBucketSetup:

  def apply(corpus: Corpus, size: DETRModelConfiguration): DETRBucketSetup = DETRBucketSetup(
    corpus = corpus,
    checkpointRoot = s"${Runs.checkpointDir}/detr-bucket/${corpus.name}-${size.name}",
    buckets = Buckets(Seq(NodeClass.Point, NodeClass.Line, NodeClass.Circle, NodeClass.Arc).map(_ -> corpus.maxNodes)),
    numLayers = size.numLayers,
    numHeads = size.numHeads,
    embedding = size.embedding
  )
