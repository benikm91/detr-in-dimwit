import dataset.Box
import dataset.Detection
import dataset.Box
import dataset.DetectionBatch
import dataset.LShapeDataset
import dataset.LShapeDataset.Split
import deepwit.checkpointing.TensorTreeCheckpointer
import deepwit.training.Monitor
import deepwit.training.tapEvery
import deepwit.attention.Head
import deepwit.attention.HeadKey
import deepwit.attention.HeadQuery
import deepwit.attention.HeadValue
import deepwit.optimizer.clipGlobalNorm
import dimwit.*
import dimwit.optimizer.Adam
import dimwit.optimizer.AdamState
import dimwit.optimizer.AdamW
import dimwit.tensor.Tensor4

/** Where [[detrTrain]] writes and [[detrEval]] reads its checkpoints. */
val CheckpointRoot = "out/detr"

private trait Batch derives Label

/** Axis of a model's parameters, flattened into one vector so that they can be counted. */
trait Parameter derives Label

case class TrainState(
    params: DETR.Params[Float32],
    optimizerState: AdamState[DETR.Params[Float32]],
    lastCost: Tensor0[Float32]
)

@main
def detrTrain(): Unit =
  dimwit.initialize()

  val numIterations = 100_000
  val batchSize = 64
  val learningRate = 3e-4f
  val weightDecay = 1e-4f

  /** Global L2 norm the gradients are rescaled to, as in the DETR paper. The set loss reassigns
    * which query is responsible for which object from step to step, so a batch that reshuffles
    * the matching produces a far larger gradient than a batch that confirms it; clipping keeps
    * those steps from undoing what the settled ones learned.
    */
  val maxGradientNorm = 0.1f

  val data = LShapeDataset.open(Axis[Width], Axis[Height], Axis[Channel], Axis[BoundingBox], Axis[Relationship])(Split.Train)
  val batches = data.objectBatches(Axis[Batch] -> batchSize)

  val optimizer = AdamW(
    Adam(learningRate = Tensor0(learningRate)),
    weightDecayFactor = Tensor0(weightDecay)
  )

  val initialParams = DETR.Params.init(
    numLayers = 3,
    numHeads = 4,
    embedding = 128,
    // The split holds at most 12 objects in a drawing, so the queries only need enough headroom
    // above that for a few of them to compete over the same object before one wins it. Every
    // query beyond that is one more slot that has to learn to stay empty.
    numQueries = 32,
    patchSize = 10,
    key = Random.Key(0)
  )

  val (flattenParams, _) = TensorTree.ravel(initialParams, Axis[Parameter])
  println(s"parameters: ${flattenParams(initialParams).shape(Axis[Parameter])}")

  val loss = HungarianLoss(VType[Float32])()

  def cost(
      imgs: Tensor4[Batch, Width, Height, Channel, Float32],
      objects: DetectionBatch[Batch, BoundingBox, Float32]
  )(params: DETR.Params[Float32]): Tensor0[Float32] =
    val model = DETR(params)
    zipvmap(Axis[Batch])(imgs, objects.box.centerX, objects.box.centerY, objects.box.width, objects.box.height, objects.label):
      case (img, centerX, centerY, width, height, label) =>
        val y = Detection(Box(centerX, centerY, width, height), label)
        val yHat = model.logits(img)
        loss(yHat, y)
    .mean

  def gradientStep(
      imgs: Tensor4[Batch, Width, Height, Channel, Float32],
      objects: DetectionBatch[Batch, BoundingBox, Float32],
      state: TrainState
  ) =
    val (lastCost, gradients) = Autodiff.valueAndGrad(cost(imgs, objects))(state.params)
    val clipped = gradients.clipGlobalNorm(Tensor0(maxGradientNorm))
    val (params, optimizerState) = optimizer.update(clipped, state.params, state.optimizerState)
    val newState = TrainState(params, optimizerState, lastCost)
    summon[TensorTree[TrainState]].map(
      state,
      [T <: Tuple, V] =>
        (labels: Labels[T]) ?=>
          (x: Tensor[T, V]) =>
            if !x.isTracer then
              dimwit.python.PyBridge.toPyTensor(x).addressable_data(0).delete()
            x
    )
    if !imgs.isTracer then
      dimwit.python.PyBridge.toPyTensor(imgs).addressable_data(0).delete()
    if !objects.box.centerX.isTracer then
      dimwit.python.PyBridge.toPyTensor(objects.box.centerX).addressable_data(0).delete
    newState
  val jitGradientStep = jitDonatingUnsafe(gradientStep)

  val checkpointer = TensorTreeCheckpointer.newIn(CheckpointRoot)
  val monitor = Monitor.default[TrainState](batchSize = batchSize, lossLens = _.lastCost.item)
  batches
    .scanLeft(TrainState(initialParams, optimizer.init(initialParams), Tensor0(-1f))):
      case (state, batch) =>
        dimwit.gc()
        jitGradientStep(batch.images, batch.target.detection, state)
    .tapEvery(10):
      case (state, step) => println(monitor.report(step, state))
    .tapEvery(1_000):
      case (state, step) =>
        checkpointer.save(state, step)
        println(s"Step $step | checkpoint saved to ${checkpointer.rootPath}")
    .drop(numIterations)
    .next()
