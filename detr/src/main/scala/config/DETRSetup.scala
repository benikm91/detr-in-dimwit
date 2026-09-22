package detr.config

import detr.*
import detr.model.*
import detr.train.*
import detr.eval.*
import dataset.Corpus
import dataset.Runs

/** Everything a training run of the detector needs to know beyond the model itself.
  *
  * A run names a corpus and a model size and everything is sized from those, so that a change to
  * how training works cannot reach one corpus and miss another. The one setting the corpus does
  * not decide by itself is how many queries to ask with, since that is a judgement about headroom
  * rather than a fact about the data — see `numQueries`.
  */
case class DETRSetup(
    corpus: Corpus,

    /** Where a run of this setup writes its checkpoints. One root per corpus and size, so that a
      * run is never read back into a model of another shape.
      */
    checkpointRoot: String,

    /** How many objects the decoder may answer with.
      *
      * A drawing holds at most `corpus.maxNodes` of them, so the queries only need enough headroom
      * above that for a few to compete over the same object before one wins it. Every query beyond
      * that is one more slot that has to learn to stay empty.
      */
    numQueries: Int,
    numLayers: Int = 3,
    numHeads: Int = 4,
    embedding: Int = 128,

    /** The same run as the transcriber's, so that the two are compared on it: 328k steps of 128
      * drawings, the last 80k of them the cooldown.
      */
    checkpointEverySamples: Int = 8_000 * 128,
    numSamples: Int = 328_000 * 128,
    batchSize: Int = 128,
    learningRate: Float = 3e-4f,

    /** Where the cosine bottoms out. Aligned with the other models. */
    finalLearningRate: Float = 0f,
    weightDecay: Float = 1e-4f,

    /** Global L2 norm the gradients are rescaled to, as in the DETR paper. The set loss reassigns
      * which query is responsible for which object from step to step, so a batch that reshuffles
      * the matching produces a far larger gradient than a batch that confirms it; clipping keeps
      * those steps from undoing what the settled ones learned.
      */
    maxGradientNorm: Float = 1f,

    /** How long the rate climbs before it holds, and how long it falls again at the end. */
    warmupSamples: Int = 1_000 * 128,
    cooldownSamples: Int = 80_000 * 128,

    seed: Int = 0
):
  require(
    numQueries > corpus.maxNodes,
    s"$numQueries queries cannot answer for a drawing of up to ${corpus.maxNodes} objects"
  )

  override def toString: String =
    s"DETRSetup(${corpus.repoId}, layers=$numLayers, heads=$numHeads, embedding=$embedding, " +
      s"queries=$numQueries, objects<=${corpus.maxNodes}, samples=$numSamples, batch=$batchSize)"

object DETRSetup:

  def apply(corpus: Corpus, size: DETR.Size): DETRSetup = DETRSetup(
    corpus = corpus,
    checkpointRoot = checkpointRoot(corpus, size),
    numQueries = queries(corpus),
    numLayers = size.numLayers,
    numHeads = size.numHeads,
    embedding = size.embedding
  )

  def checkpointRoot(corpus: Corpus, size: DETR.Size): String = s"${Runs.checkpointDir}/detr/${corpus.name}-${size.name}"

  /** About three queries per object a drawing can hold, which is the headroom the l-shape runs
    * converged with and every corpus since has kept. Fewer would leave the queries no room to
    * compete over an object before one wins it; more is more slots that have to learn to stay
    * empty.
    */
  def queries(corpus: Corpus): Int = corpus match
    case Corpus.LShape           => 32
    case Corpus.Rectilinear6to18 => 64
    case Corpus.SketchGraph      => 48
