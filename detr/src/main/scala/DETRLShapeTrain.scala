package detr.lshape

import dataset.Box
import dataset.Detection
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
import detr.*
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

/** Where [[detrLShapeTrain]] writes and [[detrLShapeEval]] reads its checkpoints. */
val CheckpointRoot = "out/detr/l-shape"

private trait Batch derives Label

/** Axis of a model's parameters, flattened into one vector so that they can be counted. */
private trait Parameter derives Label

case class TrainState(
    params: DETR.Params[Float32],
    optimizerState: LearningRateSchedulerState[DETR.Params[Float32], AdamState],
    lastCost: Tensor0[Float32]
)

/** Trains a detector on the l-shape drawings: `sbt "detr/runMain detr.lshape.detrLShapeTrain"`.
  *
  * The rectilinear run is [[detrRectilinear6to18Train]], a copy of this one differing in the
  * values at the top. Two files rather than one parameterised by a corpus, so that a run can be
  * changed without touching the other.
  */
@main
def detrLShapeTrain(): Unit =
  dimwit.initialize()

  val corpus = Corpus.LShape

  /** How many objects the decoder may answer with.
    *
    * A drawing of this corpus holds at most 12 objects, so the queries only need enough headroom
    * above that for a few of them to compete over the same object before one wins it. Every query
    * beyond that is one more slot that has to learn to stay empty.
    */
  val numQueries = 32
  require(numQueries > corpus.maxNodes, s"$numQueries queries cannot answer for a drawing of up to ${corpus.maxNodes} objects")

  val numIterations = 150_000
  val batchSize = 64
  val learningRate = 3e-4f
  val weightDecay = 1e-4f

  /** Global L2 norm the gradients are rescaled to, as in the DETR paper. The set loss reassigns
    * which query is responsible for which object from step to step, so a batch that reshuffles
    * the matching produces a far larger gradient than a batch that confirms it; clipping keeps
    * those steps from undoing what the settled ones learned.
    */
  val maxGradientNorm = 1.0f

  /** How long the rate climbs before it starts to fall. */
  val warmupSteps = 2_000

  /** Where the cosine bottoms out. Aligned with the other two models. */
  val finalLearningRate = 1e-4f

  val data = DrawingDataset.open(corpus)(Axis[Width], Axis[Height], Axis[Channel], Axis[BoundingBox], Axis[Relationship])(Split.Train)
  val batches = data.objectBatches(Axis[Batch] -> batchSize)

  /** Linear warmup into a cosine decay to nothing. Adam moves the weights by the same amount
    * whatever the gradient is, so the rate is the only thing that sets how far a step travels;
    * held constant it never shrinks and the model orbits a solution instead of settling on it.
    */
  val schedule: LearningRateSchedule =
    LinearWarmup(Tensor0(learningRate), Tensor0(warmupSteps))
      .followBy(CosineDecay(Tensor0(learningRate), Tensor0(finalLearningRate), Tensor0(numIterations - warmupSteps)))
  val optimizer = LearningRateScheduler(lr => AdamW(Adam(learningRate = lr), Tensor0(weightDecay)), schedule)

  val initialParams = DETR.Params.init(
    numLayers = 3,
    numHeads = 4,
    embedding = 128,
    numQueries = numQueries,
    patchSize = 16,
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
  val monitor = Monitor.ConcatMonitor[TrainState](List(
    Monitor.StepMonitor(),
    Monitor.LossMonitor(_.lastCost.item),
    Monitor.LearningRateMonitor(schedule),
    Monitor.PerformanceMonitor(batchSize)
  ))
  batches
    .scanLeft(TrainState(initialParams, optimizer.init(initialParams), Tensor0(-1f))):
      case (state, batch) =>
        jitGradientStep(batch.images, batch.target.detection, state)
    .tapEvery(10):
      case (state, step) => println(monitor.report(step, state))
    .tapEvery(10_000):
      case (state, step) =>
        checkpointer.save(state, step)
        println(s"Step $step | checkpoint saved to ${checkpointer.rootPath}")
    .drop(numIterations)
    .next()
