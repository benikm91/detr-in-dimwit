package detr.train

import detr.*
import detr.model.*
import detr.eval.*
import detr.config.*
import dataset.Box
import dataset.Detection
import dataset.Box
import dataset.DetectionBatch
import dataset.ObjectBatch
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
import dataset.Runs
import dimwit.*
import dimwit.Conversions.given
import dimwit.jax.Jax
import dimwit.sharding.*
import deepwit.optimizer.ConstantLearningRate
import deepwit.optimizer.CosineDecay
import deepwit.optimizer.LearningRateSchedule
import deepwit.optimizer.LearningRateScheduler
import deepwit.optimizer.LearningRateSchedulerState
import deepwit.optimizer.LinearWarmup
import dimwit.optimizer.Adam
import dimwit.optimizer.AdamState
import dimwit.optimizer.AdamW
import dimwit.tensor.Tensor4

import scala.language.implicitConversions

private trait Batch derives Label

/** Mesh axis the batch is split over. */
private trait X derives MeshLabel

/** Axis of a model's parameters, flattened into one vector so that they can be counted. */
trait Parameter derives Label

case class TrainState(
    params: DETR.Params[Float32],
    optimizerState: LearningRateSchedulerState[DETR.Params[Float32], AdamState],
    lastCost: Tensor0[Float32]
)

/** Trains a detector on the corpus its [[DETRSetup]] names.
  *
  * The batch grows with the devices, not the step count: every device takes a batch of its own and
  * the gradients are summed across them before the parameters move.
  */
def trainDetector(setup: DETRSetup): Unit =
  dimwit.initialize()
  println(s"training $setup")

  val mesh = Mesh1(MeshAxis[X] -> Jax.devices.size)
  val batchSize = setup.batchSizePerDevice * mesh.sizeOf(MeshAxis[X])
  val numSteps = setup.numSamples / batchSize
  val warmupSteps = setup.warmupSamples / batchSize
  val cooldownSteps = setup.cooldownSamples / batchSize
  val checkpointEvery = setup.checkpointEverySamples / batchSize
  println(s"$mesh on ${Jax.devices.head.platform}, $batchSize drawings per step, $numSteps steps")

  val data = DrawingDataset.open(setup.corpus)(Axis[Width], Axis[Height], Axis[Channel], Axis[BoundingBox], Axis[Relationship])(Split.Train)
  val batches = data.objectBatches(Axis[Batch] -> batchSize)

  /** Warmup, then the rate held, then a cosine cooldown over the last stretch. Adam orbits a
    * solution at a distance the rate sets, which is what the cooldown closes; holding the rate
    * until then leaves a checkpoint comparable with one from a run of another length.
    */
  val schedule: LearningRateSchedule =
    LinearWarmup(setup.learningRate, warmupSteps)
      .followBy(ConstantLearningRate(setup.learningRate, numSteps - warmupSteps - cooldownSteps))
      .followBy(CosineDecay(setup.learningRate, setup.finalLearningRate, cooldownSteps))
  val optimizer = LearningRateScheduler(lr => AdamW(Adam(learningRate = lr), setup.weightDecay), schedule)

  val initialParams = DETR.Params.init(
    numLayers = setup.numLayers,
    numHeads = setup.numHeads,
    embedding = setup.embedding,
    numQueries = setup.numQueries,
    key = Random.Key(setup.seed)
  )

  val (flattenParams, _) = TensorTree.ravel(initialParams, Axis[Parameter])
  println(s"parameters: ${flattenParams(initialParams).shape(Axis[Parameter])}")

  val loss = HungarianLoss(VType[Float32])()

  def cost(
      imgs: Tensor4[Batch |@| X, Width, Height, Channel, Float32],
      objects: DetectionBatch[Batch |@| X, BoundingBox, Float32]
  )(params: DETR.Params[Float32]): Tensor0[Float32] =
    val model = DETR(params)
    zipvmap(Axis[Batch |@| X])(imgs, objects.box.centerX, objects.box.centerY, objects.box.width, objects.box.height, objects.label):
      case (img, centerX, centerY, width, height, label) =>
        val y = Detection(Box(centerX, centerY, width, height), label)
        val yHat = model.logits(img)
        loss(yHat, y)
    .mean

  def gradientStep(
      imgs: Tensor4[Batch |@| X, Width, Height, Channel, Float32],
      objects: DetectionBatch[Batch |@| X, BoundingBox, Float32],
      state: TrainState
  ) =
    val (lastCost, gradients) = Autodiff.valueAndGrad(cost(imgs, objects))(state.params)
    val clipped = gradients.clipGlobalNorm(setup.maxGradientNorm)
    val (params, optimizerState) = optimizer.update(clipped, state.params, state.optimizerState)
    TrainState(params, optimizerState, lastCost)
  val jitGradientStep = jitDonatingUnsafe(gradientStep)

  /** The batch with every device holding its share of the drawings; the step sees one axis. */
  def shard(
      batch: dataset.Batch[Batch, Width, Height, Channel, ObjectBatch[Batch, BoundingBox]]
  ): (Tensor4[Batch |@| X, Width, Height, Channel, Float32], DetectionBatch[Batch |@| X, BoundingBox, Float32]) =
    val over = Axis[Batch] -> MeshAxis[X]
    val objects = batch.target.detection
    (
      batch.images.shard(mesh, over),
      DetectionBatch(objects.box.map(_.shard(mesh, over)), objects.label.shard(mesh, over))
    )

  val checkpointer = TensorTreeCheckpointer.newIn(setup.checkpointRoot)
  val monitor = Monitor.ConcatMonitor[TrainState](List(
    Monitor.StepMonitor(),
    Monitor.LossMonitor(_.lastCost.item),
    Monitor.LearningRateMonitor(schedule),
    Monitor.PerformanceMonitor(batchSize)
  ))
  val started = System.nanoTime
  batches
    .scanLeft(TrainState(initialParams, optimizer.init(initialParams), -1f)):
      case (state, batch) =>
        val (images, objects) = shard(batch)
        jitGradientStep(images, objects, state)
    .tapEvery(10):
      case (state, step) => println(monitor.report(step, state))
    .tapEvery(checkpointEvery):
      case (state, step) =>
        checkpointer.save(state, step)
        println(s"Step $step | checkpoint saved to ${checkpointer.rootPath}")
    .drop(numSteps)
    .next()
  Runs.noteTrainingSeconds(checkpointer.rootPath, (System.nanoTime - started) / 1_000_000_000)
