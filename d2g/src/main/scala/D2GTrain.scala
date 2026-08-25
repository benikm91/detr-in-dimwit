import dataset.Canvas
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

/** How many nodes a record is laid out in.
  *
  * A drawing of this dataset has six lines, up to six annotations, six corners and up to six
  * annotated lines, so twenty-four nodes at the most — plus somewhere for the last prediction
  * embedding to say the record has ended.
  */
val RecordNodes = 25

/** Where a training run of this model writes its checkpoints. */
val D2GCheckpointRoot = "out/d2g"

/** Axis of a batch of drawings. */
private trait Batch derives Label

case class D2GTrainState(
    params: D2G.Params[Float32],
    optimizerState: AdamState[D2G.Params[Float32]],
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
  val batchSize = 32
  val learningRate = 3e-4f
  val weightDecay = 1e-4f
  val maxGradientNorm = 0.1f
  val key = Random.Key(0)

  val nodes = Axis[Node] -> RecordNodes
  val data = LShapeDataset.open(Axis[Width], Axis[Height], Axis[Channel], Axis[Node])(Split.Train)
  var linearizeKey = Random.Key(42)
  val batches = data.batches(Axis[Batch] -> batchSize)

  val optimizer = AdamW(Adam(learningRate), weightDecay)

  val initialParams = D2G.Params.init(
    numLayers = 3,
    numHeads = 4,
    embedding = 128,
    nodes = RecordNodes,
    patchSize = 10,
    canvas = Canvas,
    key = key
  )

  val loss = RemainingNodeLoss(VType[Float32], Canvas)

  def cost(
      images: Tensor4[Batch, Width, Height, Channel, Float32],
      records: RecordBatch[Batch, Node]
  )(params: D2G.Params[Float32]): Tensor0[Float32] =
    val model = D2G(params)
    zipvmap(Axis[Batch])(images, records.nodeClass, records.xs, records.ys, records.links):
      case (image, nodeClass, xs, ys, links) =>
        val target = Record(nodeClass, xs, ys, links)
        loss(model(image, target), target)
    .mean

  def gradientStep(
      images: Tensor4[Batch, Width, Height, Channel, Float32],
      records: RecordBatch[Batch, Node],
      linearization: Key,
      state: D2GTrainState
  ) =
    val (lastCost, gradients) = Autodiff.valueAndGrad(cost(images, records.permuted(linearization, nodes)))(state.params)
    val (params, optimizerState) = optimizer.update(gradients.clipGlobalNorm(maxGradientNorm), state.params, state.optimizerState)
    D2GTrainState(params, optimizerState, lastCost)
  val jitGradientStep = jitDonatingUnsafe(gradientStep)

  val checkpointer = TensorTreeCheckpointer.newIn(D2GCheckpointRoot)
  val monitor = Monitor.default[D2GTrainState](batchSize = batchSize, lossLens = _.lastCost.item)
  batches
    .scanLeft(D2GTrainState(initialParams, optimizer.init(initialParams), Tensor0(-1f))):
      case (state, batch) =>
        val (next, forThisStep) = linearizeKey.split2()
        linearizeKey = next
        jitGradientStep(batch.images, batch.target, forThisStep, state)
    .tapEvery(10):
      case (state, step) => println(monitor.report(step, state))
    .tapEvery(1_000):
      case (state, step) =>
        checkpointer.save(state, step)
        println(s"Step $step | checkpoint saved to ${checkpointer.rootPath}")
    .drop(numIterations)
    .next()
