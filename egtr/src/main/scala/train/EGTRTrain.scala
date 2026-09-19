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
    lastCost: Tensor0[Float32]
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
  * The batch grows with the devices, not the step count: every device takes a batch of its own and
  * the gradients are summed across them before the parameters move.
  */
def trainSceneGraph(setup: EGTRSetup, detectorRun: Option[String] = None): Unit =
  dimwit.initialize()
  println(s"training $setup")

  val mesh = Mesh1(MeshAxis[X] -> Jax.devices.size)
  val batchSize = setup.batchSizePerDevice * mesh.sizeOf(MeshAxis[X])
  val numSteps = setup.numSamples / batchSize
  val warmupSteps = setup.warmupSamples / batchSize
  val checkpointEvery = setup.checkpointEverySamples / batchSize
  println(s"$mesh on ${Jax.devices.head.platform}, $batchSize drawings per step, $numSteps steps")

  val data = DrawingDataset.open(setup.corpus)(Axis[Width], Axis[Height], Axis[Channel], Axis[BoundingBox], Axis[Relationship])(Split.Train)
  val batches = data.objectBatches(Axis[Batch] -> batchSize)

  /** Linear warmup into a cosine decay to a floor. A constant rate keeps taking steps the size it
    * started with, so the model never settles.
    */
  val schedule: LearningRateSchedule =
    LinearWarmup(setup.learningRate, warmupSteps)
      .followBy(
        CosineDecay(
          setup.learningRate,
          setup.finalLearningRate,
          numSteps - warmupSteps
        )
      )
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
    EGTRTrainState(params, optimizerState, lastCost)
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

  val checkpointer = TensorTreeCheckpointer.newIn(setup.checkpointRoot)
  val monitor = Monitor.ConcatMonitor[EGTRTrainState](List(
    Monitor.StepMonitor(),
    Monitor.LossMonitor(_.lastCost.item),
    Monitor.LearningRateMonitor(schedule),
    Monitor.PerformanceMonitor(batchSize)
  ))
  val started = System.nanoTime
  batches
    .scanLeft(EGTRTrainState(initialParams, optimizer.init(initialParams), -1f)):
      case (state, batch) =>
        val (images, objects, relations) = shard(batch)
        jitGradientStep(images, objects, relations, state)
    .tapEvery(10):
      case (state, step) => println(monitor.report(step, state))
    .tapEvery(checkpointEvery):
      case (state, step) =>
        checkpointer.save(state, step)
        println(s"Step $step | checkpoint saved to ${checkpointer.rootPath}")
    .drop(numSteps)
    .next()
  Runs.noteTrainingSeconds(checkpointer.rootPath, (System.nanoTime - started) / 1_000_000_000)
