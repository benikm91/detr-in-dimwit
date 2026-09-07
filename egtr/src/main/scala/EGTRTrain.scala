import dataset.Box
import dataset.Detection
import dataset.DetectionBatch
import dataset.Corpus
import dataset.DrawingDataset
import dataset.DrawingDataset.Split
import dataset.RelationClasses
import deepwit.checkpointing.TensorTreeCheckpointer
import deepwit.training.Monitor
import deepwit.training.tapEvery
import deepwit.optimizer.clipGlobalNorm
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

import scala.language.implicitConversions

/** Axis of a batch of drawings. Named for the drawings rather than the batch because the graph
  * axes are already called after the boxes they run over.
  */
private trait Drawing derives Label

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
  */
def egtrTrain(setup: EGTRSetup, detectorRun: Option[String] = None): Unit =
  dimwit.initialize()
  println(s"training $setup")

  val data = DrawingDataset.open(setup.corpus)(Axis[Width], Axis[Height], Axis[Channel], Axis[BoundingBox], Axis[Relationship])(Split.Train)
  val batches = data.objectBatches(Axis[Drawing] -> setup.batchSize)

  /** Linear warmup into a cosine decay to a floor. A constant rate keeps taking steps the size it
    * started with, so the model never settles.
    */
  val schedule: LearningRateSchedule =
    LinearWarmup(setup.learningRate, setup.warmupSteps)
      .followBy(
        CosineDecay(
          setup.learningRate,
          setup.finalLearningRate,
          setup.numIterations - setup.warmupSteps
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
        patchSize = setup.patchSize,
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
      images: Tensor4[Drawing, Width, Height, Channel, Float32],
      objects: DetectionBatch[Drawing, BoundingBox, Float32],
      relations: Tensor4[Drawing, BoundingBox, RelatedBox, RelationClasses, Float32]
  )(params: EGTR.Params[Float32]): Tensor0[Float32] =
    val model = EGTR(params)
    zipvmap(Axis[Drawing])(images, objects.box.centerX, objects.box.centerY, objects.box.width, objects.box.height, objects.label, relations):
      case (image, centerX, centerY, width, height, label, edges) =>
        val target = SceneGraph(Detection(Box(centerX, centerY, width, height), label), edges)
        loss(model.logits(image), target)
    .mean

  def gradientStep(
      images: Tensor4[Drawing, Width, Height, Channel, Float32],
      objects: DetectionBatch[Drawing, BoundingBox, Float32],
      relations: Tensor4[Drawing, BoundingBox, RelatedBox, RelationClasses, Float32],
      state: EGTRTrainState
  ) =
    val (lastCost, gradients) = Autodiff.valueAndGrad(cost(images, objects, relations))(state.params)
    val clipped = gradients.clipGlobalNorm(setup.maxGradientNorm)
    val (params, optimizerState) = optimizer.update(clipped, state.params, state.optimizerState)
    val newState = EGTRTrainState(params, optimizerState, lastCost)
    // The donated state and the batch are dead the moment the step returns, so their device
    // buffers go with them — as in detrTrain.
    summon[TensorTree[EGTRTrainState]].map(
      state,
      [T <: Tuple, V] =>
        (labels: Labels[T]) ?=>
          (x: Tensor[T, V]) =>
            if !x.isTracer then
              dimwit.python.PyBridge.toPyTensor(x).addressable_data(0).delete()
            x
    )
    if !images.isTracer then
      dimwit.python.PyBridge.toPyTensor(images).addressable_data(0).delete()
    if !objects.box.centerX.isTracer then
      dimwit.python.PyBridge.toPyTensor(objects.box.centerX).addressable_data(0).delete
    if !relations.isTracer then
      dimwit.python.PyBridge.toPyTensor(relations).addressable_data(0).delete
    newState
  val jitGradientStep = jitDonatingUnsafe(gradientStep)

  val checkpointer = TensorTreeCheckpointer.newIn(setup.checkpointRoot)
  val monitor = Monitor.ConcatMonitor[EGTRTrainState](List(
    Monitor.StepMonitor(),
    Monitor.LossMonitor(_.lastCost.item),
    Monitor.LearningRateMonitor(schedule),
    Monitor.PerformanceMonitor(setup.batchSize)
  ))
  batches
    .scanLeft(EGTRTrainState(initialParams, optimizer.init(initialParams), -1f)):
      case (state, batch) =>
        jitGradientStep(batch.images, batch.target.detection, batch.target.relations, state)
    .tapEvery(10):
      case (state, step) => println(monitor.report(step, state))
    .tapEvery(setup.checkpointEvery):
      case (state, step) =>
        checkpointer.save(state, step)
        println(s"Step $step | checkpoint saved to ${checkpointer.rootPath}")
    .drop(setup.numIterations)
    .next()
