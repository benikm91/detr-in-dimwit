import dataset.Canvas
import dataset.MaxEdges
import dataset.MaxNodes
import dataset.LShapeDataset
import dataset.LShapeDataset.Split
import dataset.Record
import dataset.RecordBatch
import deepwit.checkpointing.TensorTreeCheckpointer
import deepwit.optimizer.clipGlobalNorm
import deepwit.training.Monitor
import deepwit.training.tapEvery
import dimwit.*
import dimwit.Conversions.given
import dimwit.optimizer.Adam
import dimwit.optimizer.AdamState
import dimwit.optimizer.AdamW
import dimwit.tensor.Tensor4

/** How many positions the nodes of a record are laid out in: one more than any drawing of this
  * dataset draws, so that the last prediction embedding has somewhere to say they have ended.
  */
val NodeSlots = MaxNodes + 1

/** The same for the relationships between them. */
val EdgeSlots = MaxEdges + 1

/** Where a training run of this model writes its checkpoints. */
val D2GCheckpointRoot = "out/d2g"

/** Axis of a batch of drawings. */
private trait Batch derives Label

case class D2GTrainState(
    params: D2G.Params[Float32],
    optimizerState: AdamState[D2G.Params[Float32]],
    linearization: Key,
    lastCost: Tensor0[Float32]
)

/** Trains a transcription model: `sbt "d2g/runMain d2gTrain"`.
  *
  * Every step draws a fresh linearization of every drawing's record, so the same drawing is seen
  * with its nodes in a different order each time it comes round — which is the point: an order
  * the loss does not commit to is an order the model cannot learn to rely on.
  */
@main
def d2gTrain(): Unit =
  dimwit.initialize()

  val numIterations = 100_000
  val batchSize = 64
  val learningRate = 3e-4f
  val weightDecay = 1e-4f
  val maxGradientNorm = 1.0f

  val nodes = Axis[Node] -> NodeSlots
  val edges = Axis[Edge] -> EdgeSlots
  val data = LShapeDataset.open(Axis[Width], Axis[Height], Axis[Channel], Axis[Node], Axis[Edge])(Split.Train)
  val batches = data.batches(Axis[Batch] -> batchSize)

  val (initKey, dataKey) = Random.Key(42).splitToTuple(2)

  val optimizer = AdamW(Adam(learningRate), weightDecay)

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

  val nodeLoss = RemainingNodeLoss(VType[Float32], Canvas)
  val edgeLoss = RemainingEdgeLoss(VType[Float32])

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
      state: D2GTrainState
  ) =
    val (nextLinearization, forThisStep) = state.linearization.split2()
    val (lastCost, gradients) = Autodiff.valueAndGrad(cost(images, records.permuted(forThisStep, nodes, edges)))(state.params)
    val (params, optimizerState) = optimizer.update(gradients.clipGlobalNorm(maxGradientNorm), state.params, state.optimizerState)
    D2GTrainState(params, optimizerState, nextLinearization, lastCost)
  val jitGradientStep = jitDonatingUnsafe(gradientStep)

  val checkpointer = TensorTreeCheckpointer.newIn(D2GCheckpointRoot)
  val monitor = Monitor.default[D2GTrainState](batchSize = batchSize, lossLens = _.lastCost.item)
  batches
    .scanLeft(D2GTrainState(initialParams, optimizer.init(initialParams), dataKey, Tensor0(-1f))):
      case (state, batch) => jitGradientStep(batch.images, batch.target, state)
    .tapEvery(100):
      case (state, step) => println(monitor.report(step, state))
    .tapEvery(1_000):
      case (state, step) =>
        checkpointer.save(state, step)
        println(s"Step $step | checkpoint saved to ${checkpointer.rootPath}")
    .drop(numIterations)
    .next()
