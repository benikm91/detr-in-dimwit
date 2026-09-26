package egtr.train

import detr.*
import detr.model.*
import detr.train.*
import detr.eval.*
import detr.config.*
import egtr.*
import egtr.model.*
import egtr.eval.*
import egtr.config.*
import dataset.Box
import dataset.Detection
import dataset.DetectionBatch
import dataset.ObjectBatch
import dataset.Corpus
import dataset.DrawingDataset
import dataset.History
import dataset.DrawingDataset.Split
import dataset.RelationClasses
import deepwit.checkpointing.TensorTreeCheckpointer
import deepwit.training.Monitor
import deepwit.training.tapEvery
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

case class EGTRTrainState(
    params: EGTR.Params[Float32],
    optimizerState: LearningRateSchedulerState[EGTR.Params[Float32], AdamState],
    loss: Tensor0[Float32]
)

/** Trains a scene graph model on the corpus its [[EGTRSetup]] names.
  *
  * The detector underneath is started from whatever `detectorRun` names, failing that from the
  * newest run under the setup's `detectorCheckpointRoot`, and failing that from scratch. Starting
  * from a trained detector is what EGTR does — the relations are read out of the detector's
  * own attention, so they have little to say until the detection is roughly right, and the
  * [[EGTRLoss]] smoothing keeps them quiet until it is. The detector is not frozen: it keeps
  * training on the joint loss.
  *
  * The batch is split over the devices: each takes the same step on its share of the drawings, and
  * the gradients are summed before the parameters move. A run on more GPUs therefore holds less per
  * device and takes the same steps — how many GPUs a run gets does not change what it learns.
  */
def trainSceneGraph(setup: EGTRSetup, detectorRun: Option[String] = None): Unit =
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

  val detector = detectorRun
    .map(TensorTreeCheckpointer(_))
    .orElse(setup.detectorCheckpointRoot.flatMap(root => TensorTreeCheckpointer.latestIn(root))) match
    case Some(checkpoints) =>
      println(s"starting from the detector of ${checkpoints.rootPath}")
      checkpoints.loadLatest[TrainState].getOrElse(sys.error(s"no checkpoint in ${checkpoints.rootPath}")).params
    case None =>
      println("starting the detector from scratch")
      DETR.Params.init(
        numLayers = setup.numLayers,
        numHeads = setup.numHeads,
        embedding = setup.embedding,
        numQueries = setup.numQueries,
        key = Random.Key(setup.seed)
      )

  val initialParams = EGTR.Params.init(
    detector = detector,
    sourceExtent = setup.sourceExtent,
    hiddenExtent = setup.hiddenExtent,
    key = Random.Key(setup.seed + 1)
  )

  val (flattenParams, _) = TensorTree.ravel(initialParams, Axis[Parameter])
  println(s"parameters: ${flattenParams(initialParams).shape(Axis[Parameter])}")

  val loss = EGTRLoss(VType[Float32], HungarianLoss(VType[Float32])())()

  def cost(
      images: Tensor4[Batch |@| X, Width, Height, Channel, Float32],
      objects: DetectionBatch[Batch |@| X, BoundingBox, Float32],
      relations: Tensor4[Batch |@| X, BoundingBox, RelatedBox, RelationClasses, Float32]
  )(params: EGTR.Params[Float32]): Tensor0[Float32] =
    val model = EGTR(params)
    zipvmap(Axis[Batch |@| X])(images, objects.box.centerX, objects.box.centerY, objects.box.width, objects.box.height, objects.label, relations):
      case (image, centerX, centerY, width, height, label, edges) =>
        val target = SceneGraph(Detection(Box(centerX, centerY, width, height), label), edges)
        loss(model.logits(image), target)
    .mean

  def gradientStep(
      images: Tensor4[Batch |@| X, Width, Height, Channel, Float32],
      objects: DetectionBatch[Batch |@| X, BoundingBox, Float32],
      relations: Tensor4[Batch |@| X, BoundingBox, RelatedBox, RelationClasses, Float32],
      state: EGTRTrainState
  ) =
    val (lastCost, gradients) = Autodiff.valueAndGrad(cost(images, objects, relations))(state.params)
    val clipped = gradients.clipGlobalNorm(setup.maxGradientNorm)
    val (params, optimizerState) = optimizer.update(clipped, state.params, state.optimizerState)
    EGTRTrainState(params, optimizerState, state.loss * 0.99f + lastCost * 0.01f)
  val jitGradientStep = jitDonatingUnsafe(gradientStep)

  /** The batch with every device holding its share of the drawings; the step sees one axis. */
  def shard(
      batch: dataset.Batch[Batch, Width, Height, Channel, ObjectBatch[Batch, BoundingBox]]
  ): (
      Tensor4[Batch |@| X, Width, Height, Channel, Float32],
      DetectionBatch[Batch |@| X, BoundingBox, Float32],
      Tensor4[Batch |@| X, BoundingBox, RelatedBox, RelationClasses, Float32]
  ) =
    val over = Axis[Batch] -> MeshAxis[X]
    val objects = batch.target.detection
    (
      batch.images.shard(mesh, over),
      DetectionBatch(objects.box.map(_.shard(mesh, over)), objects.label.shard(mesh, over)),
      batch.target.relations.shard(mesh, over)
    )

  /** The run's own folder, continued where it already holds checkpoints: a job that runs out
    * of time puts itself back in the queue, and what starts again carries on from the newest
    * one. The schedule rides along in the optimizer's state, so the cooldown still falls where
    * it was always going to. Writing into a folder that is already there is what `overwrite`
    * allows; it adds checkpoints and removes none.
    */
  val checkpointer = TensorTreeCheckpointer.latestIn(setup.checkpointRoot) match
    case Some(started) => TensorTreeCheckpointer(started.rootPath, overwrite = true)
    case None          => TensorTreeCheckpointer.newIn(setup.checkpointRoot)
  val taken = checkpointer.iterations.maxOption.getOrElse(0)
  val initialState = if taken == 0 then EGTRTrainState(initialParams, optimizer.init(initialParams), 0f)
    else checkpointer.load[EGTRTrainState](taken).getOrElse(sys.error(s"checkpoint $taken of ${checkpointer.rootPath} will not load"))
  if taken > 0 then println(s"continuing ${checkpointer.rootPath} from step $taken")
  val history = History()
  val monitor = Monitor.ConcatMonitor[EGTRTrainState](List(
    Monitor.StepMonitor(),
    Monitor.LossMonitor(_.loss.item),
    Monitor.LearningRateMonitor(schedule),
    Monitor.PerformanceMonitor(batchSize)
  ))

  val started = System.nanoTime
  batches
    .scanLeft(initialState):
      case (state, batch) =>
        val (images, objects, relations) = shard(batch)
        jitGradientStep(images, objects, relations, state)
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
