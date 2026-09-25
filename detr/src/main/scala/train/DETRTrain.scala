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
import dataset.History
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
    loss: Tensor0[Float32]
)

/** Trains a detector on the corpus its [[DETRSetup]] names.
  *
  * The batch is split over the devices: each takes the same step on its share of the drawings, and
  * the gradients are summed before the parameters move. A run on more GPUs therefore holds less per
  * device and takes the same steps — how many GPUs a run gets does not change what it learns.
  */
def trainDetector(setup: DETRSetup): Unit =
  dimwit.initialize()
  println(s"training $setup")

  val mesh = Mesh1(MeshAxis[X] -> Jax.devices.size)
  val batchSize = setup.batchSize
  require(batchSize % mesh.sizeOf(MeshAxis[X]) == 0, s"a batch of $batchSize does not split over $mesh")
  val numTotalSteps = setup.numSamples / batchSize
  val warmupSteps = setup.warmupSamples / batchSize
  val cooldownSteps = setup.cooldownSamples / batchSize
  val checkpointEvery = setup.checkpointEverySamples / batchSize
  require(numTotalSteps % checkpointEvery == 0, s"$numTotalSteps steps do not end on a checkpoint, which is where the cooldown ends")
  println(s"$mesh on ${Jax.devices.head.platform}, $batchSize drawings per step, $numTotalSteps steps")

  val data = DrawingDataset.open(setup.corpus)(Axis[Width], Axis[Height], Axis[Channel], Axis[BoundingBox], Axis[Relationship])(Split.Train)
  val batches = data.objectBatches(Axis[Batch] -> batchSize)

  /** Warmup, then the rate held, then a cosine cooldown over the last stretch. Adam orbits a
    * solution at a distance the rate sets, which is what the cooldown closes; holding the rate
    * until then leaves a checkpoint comparable with one from a run of another length.
    */
  val schedule: LearningRateSchedule =
    LinearWarmup(setup.learningRate, warmupSteps)
      .followBy(ConstantLearningRate(setup.learningRate, numTotalSteps - warmupSteps - cooldownSteps))
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
    TrainState(params, optimizerState, state.loss * 0.99f + lastCost * 0.01f)
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

  /** The run's own folder, continued where it already holds checkpoints: a job that runs out
    * of time puts itself back in the queue, and what starts again carries on from the newest
    * one. The schedule rides along in the optimizer's state, so the cooldown still falls where
    * it was always going to.
    */
  val checkpointer = TensorTreeCheckpointer.latestIn(setup.checkpointRoot).getOrElse(TensorTreeCheckpointer.newIn(setup.checkpointRoot))
  val taken = checkpointer.iterations.maxOption.getOrElse(0)
  val initialState = if taken == 0 then TrainState(initialParams, optimizer.init(initialParams), 0f)
    else checkpointer.load[TrainState](taken).getOrElse(sys.error(s"checkpoint $taken of ${checkpointer.rootPath} will not load"))
  if taken > 0 then println(s"continuing ${checkpointer.rootPath} from step $taken")
  val history = History()
  val monitor = Monitor.ConcatMonitor[TrainState](List(
    Monitor.StepMonitor(),
    Monitor.LossMonitor(_.loss.item),
    Monitor.LearningRateMonitor(schedule),
    Monitor.PerformanceMonitor(batchSize)
  ))

  val started = System.nanoTime
  batches
    .scanLeft(initialState):
      case (state, batch) =>
        val (images, objects) = shard(batch)
        jitGradientStep(images, objects, state)
    .tapEvery(10):
      case (state, step) => println(monitor.report(taken + step, state))
    .tapEvery(checkpointEvery):
      case (state, step) =>
        checkpointer.save(state, taken + step)
        history.add(taken + step, state.loss.item)
        println(s"Step ${taken + step} | checkpoint saved to ${checkpointer.rootPath}")
    .drop(numTotalSteps - taken)
    .next()
  Runs.noteTrainingSeconds(checkpointer.rootPath, (System.nanoTime - started) / 1_000_000_000)
