import dataset.Corpus

/** Everything a training run of the scene graph model needs to know beyond the model itself.
  *
  * The same reasoning as [[D2GSetup]] and [[DETRSetup]]: a corpus is picked once and everything
  * sized from it. What this one adds is where the detector underneath comes from — see
  * `detectorCheckpointRoot`.
  */
case class EGTRSetup(
    corpus: Corpus,

    /** Where a run of this setup writes its checkpoints. One root per corpus. */
    checkpointRoot: String,

    /** How many objects the decoder may answer with. As in [[DETRSetup]]. */
    numQueries: Int,

    /** Where to look for a detector to start from, or `None` to start from scratch.
      *
      * Starting from a trained detector is what the paper does — the relations are read out of the
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
      * the heads read. The paper keeps both at the detector's embedding width.
      */
    sourceExtent: Int = 128,
    hiddenExtent: Int = 128,
    patchSize: Int = 16,
    numIterations: Int = 150_000,
    batchSize: Int = 64,
    learningRate: Float = 3e-4f,

    /** Where the cosine bottoms out. Aligned with the other models. */
    finalLearningRate: Float = 1e-4f,
    weightDecay: Float = 1e-4f,
    maxGradientNorm: Float = 1f,

    /** How long the rate climbs before it starts to fall. */
    warmupSteps: Int = 2_000,

    /** Above which score a relation counts as predicted.
      *
      * A record is a set of relationships, not a ranking of them, so a relation has to be either
      * claimed or not before the graph can be compared with the one the drawing holds. The scene
      * graph literature avoids the question by reporting `R@k` at a fixed `k`; a record cannot,
      * since predicting too many relationships has to cost something.
      */
    relationThreshold: Float = 0.5f,
    checkpointEvery: Int = 10_000,
    seed: Int = 0
):
  require(
    numQueries > corpus.maxNodes,
    s"$numQueries queries cannot answer for a drawing of up to ${corpus.maxNodes} objects"
  )

  override def toString: String =
    s"EGTRSetup(${corpus.repoId}, layers=$numLayers, heads=$numHeads, embedding=$embedding, " +
      s"queries=$numQueries, objects<=${corpus.maxNodes}, iterations=$numIterations, batch=$batchSize, " +
      s"detector=${detectorCheckpointRoot.getOrElse("from scratch")})"
