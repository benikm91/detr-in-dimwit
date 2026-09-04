import dataset.Corpus

/** Everything a training run of the transcription model needs to know beyond the model itself.
  *
  * The corpora differ in more than their pictures: a rectilinear part holds nearly twice the nodes
  * an L does, so the sequence the model reads and writes is nearly twice as long, and the model
  * that suits one need not suit the other. Holding the choice in one value means a corpus is picked
  * once and everything sized from it — the record slots, the loader, and where the run writes —
  * rather than in as many places as there are things that depend on it.
  *
  * The defaults are the settings the l-shape runs converged under, so a new corpus starts from
  * those and changes only what it has reason to.
  */
case class D2GSetup(
    corpus: Corpus,

    /** Where a run of this setup writes its checkpoints. One root per corpus, so that a run on one
      * is never read back as a run on the other — the shapes would not even load.
      */
    checkpointRoot: String,
    numLayers: Int = 3,
    numHeads: Int = 4,
    embedding: Int = 128,
    patchSize: Int = 16,
    numIterations: Int = 150_000,
    batchSize: Int = 64,
    learningRate: Float = 3e-4f,

    /** Where the cosine bottoms out rather than reaching nothing.
      *
      * A record is written in two stages and the relationships can only be learned once the nodes
      * they name are right, so they are learned late — decaying the rate to zero takes the rate
      * away exactly when that half of the model still needs it. Measured at 100k: decayed to zero,
      * the relationships fall from 91.5% to 89.6% and whole records from 73.4% to 50.2%, while the
      * nodes improve. A floor keeps the late half learning.
      */
    finalLearningRate: Float = 1e-4f,
    weightDecay: Float = 1e-4f,
    maxGradientNorm: Float = 1f,

    /** How long the rate climbs before it starts to fall. Adam's own step is the same size whatever
      * the gradient is, so a fresh model with a meaningless gradient would otherwise take full
      * sized steps in an arbitrary direction.
      */
    warmupSteps: Int = 2_000,

    /** Whether equation 4's pass-through term is included — the one that keeps a taken node's own
      * embedding carrying it while the prediction embedding beside it becomes a different node.
      * Off by default: measured on l-shapes it costs 3.2 points of whole records and roughly
      * doubles the time to converge.
      */
    withPassThrough: Boolean = false,

    /** How hard the two prediction tokens of a slot are jogged apart, as a fraction of the
      * embedding's own size.
      *
      * They are otherwise the same token at the same position and cannot see each other, so this
      * noise is the only thing that can make them answer with different remaining nodes — which is
      * what the loss asks of them, and what stops the model learning to answer only with the node
      * it finds easiest. Too small and they agree and the loss is inert; too large and it drowns
      * the positional encoding they need in order to know which slot they answer for.
      */
    predictionNoise: Float = 0.15f,

    /** What agreeing costs the two prediction tokens of a slot, in nats.
      *
      * Each token is charged only for the remaining node it chose itself, so nothing ever hands one
      * of them a node it did not pick — this is what pushes them apart instead. It is a cost they
      * may always choose to pay, which matters at the end of a record where two different answers
      * can be impossible: paying it and committing has to stay cheaper than splitting the
      * difference between two nodes, which costs about 4 ln 2 = 2.77. Below that, and above what
      * naming a second-best node costs, is the window.
      */
    separation: Float = 1f,
    checkpointEvery: Int = 10_000,
    seed: Int = 42
):

  /** How many positions the nodes of a record are laid out in: one more than any drawing of the
    * corpus draws, so that the last prediction embedding has somewhere to say they have ended.
    */
  val nodeSlots: Int = corpus.maxNodes + 1

  /** The same for the relationships between them. */
  val edgeSlots: Int = corpus.maxEdges + 1

  override def toString: String =
    s"D2GSetup(${corpus.repoId}, layers=$numLayers, heads=$numHeads, embedding=$embedding, " +
      s"nodes=$nodeSlots, edges=$edgeSlots, iterations=$numIterations, batch=$batchSize)"
