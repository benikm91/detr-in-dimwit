package d2g.rectilinear6to18

import dataset.Canvas
import dataset.Corpus
import dataset.DrawingDataset
import dataset.DrawingDataset.Split
import dataset.Record
import dataset.RecordBatch
import deepwit.checkpointing.TensorTreeCheckpointer
import deepwit.optimizer.clipGlobalNorm
import deepwit.training.Monitor
import deepwit.training.tapEvery
import d2g.*
import dimwit.*
import dimwit.Conversions.given
import deepwit.optimizer.CosineDecay
import deepwit.optimizer.LearningRateSchedule
import deepwit.optimizer.LearningRateScheduler
import deepwit.optimizer.LearningRateSchedulerState
import deepwit.optimizer.LinearWarmup
import dimwit.optimizer.Adam
import dimwit.optimizer.AdamState
import dimwit.optimizer.AdamW
import dimwit.tensor.Tensor4

/** How many positions the nodes of a record are laid out in: one more than any drawing of the
  * rectilinear corpus draws, so that the last prediction embedding has somewhere to say they have
  * ended.
  */
val NodeSlots = Corpus.Rectilinear6to18.maxNodes + 1

/** The same for the relationships between them. */
val EdgeSlots = Corpus.Rectilinear6to18.maxEdges + 1

/** Where a training run of this model writes its checkpoints. */
val CheckpointRoot = "out/d2g/rectilinear-6to18"

/** Axis of a batch of drawings. */
private trait Batch derives Label

/** Axis of a model's parameters, flattened into one vector so that they can be counted. */
private trait Parameter derives Label

case class TrainState(
    params: D2G.Params[Float32],
    optimizerState: LearningRateSchedulerState[D2G.Params[Float32], AdamState],
    linearization: Key,
    lastCost: Tensor0[Float32]
)

/** Trains a transcription model on the rectilinear parts of six to eighteen lines:
  * `sbt "d2g/runMain d2g.rectilinear6to18.d2gRectilinear6to18Train"`.
  *
  * A copy of [[d2gLShapeTrain]] differing in the values at the top. The model is the size the
  * l-shape run uses, deliberately: the corpus is what changes here, and a model that changed with
  * it would leave the two unable to be compared. Only the record slots grow, and they grow because
  * the corpus says so — 22 nodes and 22 relationships against the L's 12, so the sequence read and
  * written is nearly twice as long.
  */
@main
def d2gRectilinear6to18Train(): Unit =
  dimwit.initialize()

  val corpus = Corpus.Rectilinear6to18

  val numIterations = 150_000
  val batchSize = 64
  val learningRate = 3e-4f
  val weightDecay = 1e-4f
  val maxGradientNorm = 1.0f

  /** How long the rate climbs before it starts to fall. Adam's own step is the same size whatever
    * the gradient is, so a fresh model with a meaningless gradient would otherwise take full sized
    * steps in an arbitrary direction.
    */
  val warmupSteps = 2_000

  /** Where the cosine bottoms out rather than reaching nothing.
    *
    * A record is written in two stages and the relationships can only be learned once the nodes
    * they name are right, so they are learned late — decaying the rate to zero takes the rate away
    * exactly when that half of the model still needs it. Measured at 100k: decayed to zero, the
    * relationships fall from 91.5% to 89.6% and whole records from 73.4% to 50.2%, while the nodes
    * improve. A floor keeps the late half learning.
    */
  val finalLearningRate = 1e-4f

  /** Whether equation 4's pass-through term is included — the one that keeps a taken node's own
    * embedding carrying it while the prediction embedding beside it becomes a different node.
    * Turned off to measure what it is worth: it costs 3.2 points of whole records and roughly
    * doubles the time to converge.
    */
  val withPassThrough = false

  val nodes = Axis[Node] -> NodeSlots
  val edges = Axis[Edge] -> EdgeSlots
  val data = DrawingDataset.open(corpus)(Axis[Width], Axis[Height], Axis[Channel], Axis[Node], Axis[Edge])(Split.Train)
  val batches = data.batches(Axis[Batch] -> batchSize)

  val (initKey, dataKey) = Random.Key(42).splitToTuple(2)

  /** Linear warmup into a cosine decay to a floor.
    *
    * Clipping cannot set the step size here — Adam normalises per parameter, so a clipped
    * gradient and an unclipped one move the weights by the same 0.2% of their norm — which leaves
    * the rate as the only thing that can. Held constant it never shrinks, so the model keeps
    * taking full sized steps long after it has found a solution and can only orbit one; decayed
    * it can actually settle on it.
    */
  val schedule: LearningRateSchedule =
    LinearWarmup(Tensor0(learningRate), Tensor0(warmupSteps))
      .followBy(CosineDecay(Tensor0(learningRate), Tensor0(finalLearningRate), Tensor0(numIterations - warmupSteps)))
  val optimizer = LearningRateScheduler(lr => AdamW(Adam(learningRate = lr), Tensor0(weightDecay)), schedule)

  val initialParams = D2G.Params.init(
    numLayers = 3,
    numHeads = 4,
    embedding = 128,
    nodes = NodeSlots,
    edges = EdgeSlots,
    patchSize = 16,
    canvas = Canvas,
    key = initKey
  )

  val (flattenParams, _) = TensorTree.ravel(initialParams, Axis[Parameter])
  println(s"parameters: ${flattenParams(initialParams).shape(Axis[Parameter])}")

  val nodeLoss = RemainingNodeLoss(VType[Float32], Canvas, withPassThrough)
  val edgeLoss = RemainingEdgeLoss(VType[Float32], withPassThrough)

  def cost(
      images: Tensor4[Batch, Width, Height, Channel, Float32],
      records: RecordBatch[Batch, Node, Edge]
  )(params: D2G.Params[Float32]): Tensor0[Float32] =
    val model = D2G(params)
    zipvmap(Axis[Batch])(images, records.nodeClass, records.startX, records.startY, records.endX, records.endY, records.edgeClass, records.subject, records.obj):
      case (image, nodeClass, startX, startY, endX, endY, edgeClass, subject, obj) =>
        val target = Record(nodeClass, startX, startY, endX, endY, edgeClass, subject, obj)
        val scored = model(image, target)
        nodeLoss(scored.nodes, target.nodes) + edgeLoss(scored.edges, target.edges)
    .mean

  def gradientStep(
      images: Tensor4[Batch, Width, Height, Channel, Float32],
      records: RecordBatch[Batch, Node, Edge],
      state: TrainState
  ) =
    val (nextLinearization, forThisStep) = state.linearization.split2()
    val (lastCost, gradients) = Autodiff.valueAndGrad(cost(images, records.permuted(forThisStep, nodes, edges)))(state.params)
    val (params, optimizerState) = optimizer.update(gradients.clipGlobalNorm(maxGradientNorm), state.params, state.optimizerState)
    TrainState(params, optimizerState, nextLinearization, lastCost)
  val jitGradientStep = jitDonatingUnsafe(gradientStep)

  val checkpointer = TensorTreeCheckpointer.newIn(CheckpointRoot)
  val monitor = Monitor.ConcatMonitor[TrainState](List(
    Monitor.StepMonitor(),
    Monitor.LossMonitor(_.lastCost.item),
    Monitor.LearningRateMonitor(schedule),
    Monitor.PerformanceMonitor(batchSize)
  ))
  batches
    .scanLeft(TrainState(initialParams, optimizer.init(initialParams), dataKey, Tensor0(-1f))):
      case (state, batch) => jitGradientStep(batch.images, batch.target, state)
    .tapEvery(100):
      case (state, step) => println(monitor.report(step, state))
    .tapEvery(10_000):
      case (state, step) =>
        checkpointer.save(state, step)
        println(s"Step $step | checkpoint saved to ${checkpointer.rootPath}")
    .drop(numIterations)
    .next()
