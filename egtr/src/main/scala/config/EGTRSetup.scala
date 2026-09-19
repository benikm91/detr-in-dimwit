package egtr.config

import detr.*
import detr.model.*
import detr.train.*
import detr.eval.*
import detr.config.*
import egtr.*
import egtr.model.*
import egtr.train.*
import egtr.eval.*
import dataset.Corpus
import dataset.Runs

/** Everything a training run of the scene graph model needs to know beyond the model itself.
  *
  * The same reasoning as [[D2GSetup]] and [[DETRSetup]]: a run names a corpus and a model size and
  * everything is sized from those. What this one adds is where the detector underneath comes from
  * — see `detectorCheckpointRoot`.
  */
case class EGTRSetup(
    corpus: Corpus,

    /** Where a run of this setup writes its checkpoints. One root per corpus and size. */
    checkpointRoot: String,

    /** How many objects the decoder may answer with. As in [[DETRSetup]]. */
    numQueries: Int,

    /** Where to look for a detector to start from, or `None` to start from scratch.
      *
      * Starting from a trained detector is what EGTR does — the relations are read out of the
      * detector's own attention, so they have little to say until the detection is roughly right.
      * It also makes the model incomparable with one trained from nothing, since it has been shown
      * the corpus twice. Which of the two is wanted depends on what the run is for, so it is said
      * here rather than assumed.
      *
      * sbt forks a `runMain` from the base directory of the project it belongs to, so the runs of
      * the detector sit under the detr project rather than next to this one's.
      */
    detectorCheckpointRoot: Option[String] = None,
    numLayers: Int = 3,
    numHeads: Int = 4,
    embedding: Int = 128,

    /** One source projects a query into this, and a pair of queries into twice it, which is what
      * the heads read. EGTR keeps both at the detector's embedding width.
      */
    sourceExtent: Int = 128,
    hiddenExtent: Int = 128,

    numSamples: Int = 200_000 * 64,
    batchSizePerDevice: Int = 64,
    learningRate: Float = 3e-4f,

    /** Where the cosine bottoms out. Aligned with the other models. */
    finalLearningRate: Float = 1e-4f,
    weightDecay: Float = 1e-4f,
    maxGradientNorm: Float = 1f,

    /** How long the rate climbs before it starts to fall. */
    warmupSamples: Int = 2_000 * 64,

    checkpointEverySamples: Int = 10_000 * 64,
    seed: Int = 0
):
  require(
    numQueries > corpus.maxNodes,
    s"$numQueries queries cannot answer for a drawing of up to ${corpus.maxNodes} objects"
  )

  override def toString: String =
    s"EGTRSetup(${corpus.repoId}, layers=$numLayers, heads=$numHeads, embedding=$embedding, " +
      s"queries=$numQueries, objects<=${corpus.maxNodes}, samples=$numSamples, batch=$batchSizePerDevice per device, " +
      s"detector=${detectorCheckpointRoot.getOrElse("from scratch")})"

object EGTRSetup:

  /** From scratch, which is what makes a run comparable with a transcription model given nothing
    * but the drawings. The queries are the detector's: this model is that detector and a head on
    * it.
    */
  def apply(corpus: Corpus, size: EGTR.Size): EGTRSetup = EGTRSetup(
    corpus = corpus,
    checkpointRoot = s"${Runs.checkpointDir}/egtr/${corpus.name}-${size.name}",
    numQueries = DETRSetup.queries(corpus),
    numLayers = size.numLayers,
    numHeads = size.numHeads,
    embedding = size.embedding
  )
