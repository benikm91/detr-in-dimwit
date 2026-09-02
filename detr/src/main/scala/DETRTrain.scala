import dataset.Box
import dataset.Detection
import dataset.Box
import dataset.DetectionBatch
import dataset.Corpus
import dataset.DrawingDataset
import dataset.DrawingDataset.Split
import deepwit.checkpointing.TensorTreeCheckpointer
import deepwit.training.Monitor
import deepwit.training.tapEvery
import deepwit.attention.Head
import deepwit.attention.HeadKey
import deepwit.attention.HeadQuery
import deepwit.attention.HeadValue
import deepwit.optimizer.clipGlobalNorm
import dimwit.*
import deepwit.optimizer.CosineDecay
import deepwit.optimizer.LearningRateSchedule
import deepwit.optimizer.LearningRateScheduler
import deepwit.optimizer.LearningRateSchedulerState
import deepwit.optimizer.LinearWarmup
import dimwit.optimizer.Adam
import dimwit.optimizer.AdamState
import dimwit.optimizer.AdamW
import dimwit.tensor.Tensor4

private trait Batch derives Label

/** Axis of a model's parameters, flattened into one vector so that they can be counted. */
trait Parameter derives Label

case class TrainState(
    params: DETR.Params[Float32],
    optimizerState: LearningRateSchedulerState[DETR.Params[Float32], AdamState],
    lastCost: Tensor0[Float32]
)

/** Trains a detector on the corpus its [[DETRSetup]] names. */
def detrTrain(setup: DETRSetup): Unit =
  dimwit.initialize()
  println(s"training $setup")

  val data = DrawingDataset.open(setup.corpus)(Axis[Width], Axis[Height], Axis[Channel], Axis[BoundingBox], Axis[Relationship])(Split.Train)
  val batches = data.objectBatches(Axis[Batch] -> setup.batchSize)

  /** Linear warmup into a cosine decay to a floor. Adam moves the weights by the same amount
    * whatever the gradient is, so the rate is the only thing that sets how far a step travels;
    * held constant it never shrinks and the model orbits a solution instead of settling on it.
    */
  val schedule: LearningRateSchedule =
    LinearWarmup(Tensor0(setup.learningRate), Tensor0(setup.warmupSteps))
      .followBy(
        CosineDecay(
          Tensor0(setup.learningRate),
          Tensor0(setup.finalLearningRate),
          Tensor0(setup.numIterations - setup.warmupSteps)
        )
      )
  val optimizer = LearningRateScheduler(lr => AdamW(Adam(learningRate = lr), Tensor0(setup.weightDecay)), schedule)

  val initialParams = DETR.Params.init(
    numLayers = setup.numLayers,
    numHeads = setup.numHeads,
    embedding = setup.embedding,
    numQueries = setup.numQueries,
    patchSize = setup.patchSize,
    key = Random.Key(setup.seed)
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
    val clipped = gradients.clipGlobalNorm(Tensor0(setup.maxGradientNorm))
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

  val checkpointer = TensorTreeCheckpointer.newIn(setup.checkpointRoot)
  val monitor = Monitor.ConcatMonitor[TrainState](List(
    Monitor.StepMonitor(),
    Monitor.LossMonitor(_.lastCost.item),
    Monitor.LearningRateMonitor(schedule),
    Monitor.PerformanceMonitor(setup.batchSize)
  ))
  batches
    .scanLeft(TrainState(initialParams, optimizer.init(initialParams), Tensor0(-1f))):
      case (state, batch) =>
        jitGradientStep(batch.images, batch.target.detection, state)
    .tapEvery(10):
      case (state, step) => println(monitor.report(step, state))
    .tapEvery(setup.checkpointEvery):
      case (state, step) =>
        checkpointer.save(state, step)
        println(s"Step $step | checkpoint saved to ${checkpointer.rootPath}")
    .drop(setup.numIterations)
    .next()
