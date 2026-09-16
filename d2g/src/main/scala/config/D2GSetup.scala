package d2g.config

import d2g.*
import d2g.model.*
import d2g.train.*
import d2g.eval.*
import dataset.Corpus
import dataset.Runs

/** Everything a training run of the transcription model needs to know beyond the model itself.
  *
  * The corpora differ in more than their pictures: a rectilinear part holds nearly twice the nodes
  * an L does, so the sequence the model reads and writes is nearly twice as long, and the model
  * that suits one need not suit the other. Holding the choice in one value means a corpus is picked
  * once and everything sized from it — the record slots, the loader, and where the run writes —
  * rather than in as many places as there are things that depend on it.
  *
  * A run names a corpus and a model size; everything else is fixed.
  */
case class D2GSetup(
    corpus: Corpus,

    /** Where a run of this setup writes its checkpoints. One root per corpus and size, so that a
      * run is never read back into a model of another shape — it would not even load.
      */
    checkpointRoot: String,
    numLayers: Int = 3,
    numHeads: Int = 4,
    embedding: Int = 128,
    patchSize: Int = 16,
    numIterations: Int = 200_000,
    batchSize: Int = 64,
    learningRate: Float = 3e-4f,

    /** Where the cosine bottoms out rather than reaching nothing. A record is written in two
      * stages and the relationships can only be learned once the nodes they name are right, so a
      * floor keeps a rate for the half that is still learning at the end.
      */
    finalLearningRate: Float = 1e-4f,
    weightDecay: Float = 1e-4f,
    maxGradientNorm: Float = 1f,

    /** How long the rate climbs before it starts to fall. Adam's own step is the same size whatever
      * the gradient is, so a fresh model with a meaningless gradient would otherwise take full
      * sized steps in an arbitrary direction.
      */
    warmupSteps: Int = 2_000,

    /** How many query vectors the model learns in all.
      *
      * A slot answers with a pair of them while training, drawn afresh every step, so the pool is
      * what the model must be able to answer with and the pair is what a step costs. Every pair is
      * asked to name different nodes and every pair comes round, so the whole pool ends up
      * distinct — a ranking as deep as the pool for the price of two tokens.
      */
    queryPool: Int = 6,

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

object D2GSetup:

  def apply(corpus: Corpus, size: D2G.Size): D2GSetup = D2GSetup(
    corpus = corpus,
    checkpointRoot = s"${Runs.checkpointDir}/d2g/${corpus.name}-${size.name}",
    numLayers = size.numLayers,
    numHeads = size.numHeads,
    embedding = size.embedding
  )
