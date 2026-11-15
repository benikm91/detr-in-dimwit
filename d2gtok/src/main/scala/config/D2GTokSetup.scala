package d2gtok.config

import d2g.floorplan.RelationshipOrder
import d2tok.config.D2TokModelConfiguration
import d2tok.floorplan.config.D2TokSetup
import dataset.Runs
import syntheticFloorplansDataset.FloorplanCorpus

/** Setup for a D2GTok experiment on the synthetic floor plans. */
case class D2GTokSetup(
    /** The d2tok run that writes the nodes: its corpus, size and canvas, the words it writes in, and
      * the order it writes the nodes in, which the relationships are trained on.
      */
    nodeWriter: D2TokSetup,
    /** Where the newest checkpoint of that run is. */
    nodesFrom: String,

    queryPool: Int = 6,
    /** Whether the relationships of a record are written down in any order or kind by kind. */
    relationshipOrder: RelationshipOrder = RelationshipOrder.Free,

    // Train configuration
    checkpointRoot: String,
    /** 200k steps of 512 plans, about 228 passes over the plain training split and 102 over the
      * furnished one: 10k steps of warmup, 150k at the full learning rate and a 40k cooldown, with a
      * checkpoint every 10k.
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

  val corpus: FloorplanCorpus = nodeWriter.corpus
  val nodeSlots: Int = corpus.maxNodes
  val edgeSlots: Int = corpus.maxRelationships + 1 // One more due to the end prediction

  override def toString: String = s"D2GTokSetup(${corpus.repoId}, nodes by $nodeWriter from $nodesFrom, relationship order=$relationshipOrder, edges=$edgeSlots, samples=$numSamples, batch=$batchSize)"

object D2GTokSetup:

  def apply(corpus: FloorplanCorpus, size: D2TokModelConfiguration): D2GTokSetup = D2GTokSetup(
    nodeWriter = D2TokSetup(corpus, size),
    nodesFrom = s"${sys.env.getOrElse("NODES_DIR", Runs.checkpointDir)}/d2tok/${corpus.name}-${size.name}",
    checkpointRoot = s"${Runs.checkpointDir}/d2gtok/${corpus.name}-${size.name}"
  )
