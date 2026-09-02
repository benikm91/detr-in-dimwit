import dataset.Corpus

/** Everything a training run of the detector needs to know beyond the model itself.
  *
  * The same reasoning as [[D2GSetup]]: a corpus is picked once and everything sized from it, so
  * that a change to how training works cannot reach one corpus and miss another. The one setting
  * the corpus does not decide by itself is how many queries to ask with, since that is a judgement
  * about headroom rather than a fact about the data — see `numQueries`.
  */
case class DETRSetup(
    corpus: Corpus,

    /** Where a run of this setup writes its checkpoints. One root per corpus. */
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
    patchSize: Int = 16,
    numIterations: Int = 150_000,
    batchSize: Int = 64,
    learningRate: Float = 3e-4f,

    /** Where the cosine bottoms out. Aligned with the other models. */
    finalLearningRate: Float = 1e-4f,
    weightDecay: Float = 1e-4f,

    /** Global L2 norm the gradients are rescaled to, as in the DETR paper. The set loss reassigns
      * which query is responsible for which object from step to step, so a batch that reshuffles
      * the matching produces a far larger gradient than a batch that confirms it; clipping keeps
      * those steps from undoing what the settled ones learned.
      */
    maxGradientNorm: Float = 1f,

    /** How long the rate climbs before it starts to fall. */
    warmupSteps: Int = 2_000,
    checkpointEvery: Int = 10_000,
    seed: Int = 0
):
  require(
    numQueries > corpus.maxNodes,
    s"$numQueries queries cannot answer for a drawing of up to ${corpus.maxNodes} objects"
  )

  override def toString: String =
    s"DETRSetup(${corpus.repoId}, layers=$numLayers, heads=$numHeads, embedding=$embedding, " +
      s"queries=$numQueries, objects<=${corpus.maxNodes}, iterations=$numIterations, batch=$batchSize)"
