package egtr.lshape

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
import detr.*
import detr.lshape.TrainState as DetectorState
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
import egtr.*

/** Where [[egtrLShapeTrain]] writes and [[egtrLShapeEval]] reads its checkpoints. */
val CheckpointRoot = "out/egtr/l-shape"

/** Where [[egtrLShapeTrain]] looks for a detector to start from, or `None` to start from scratch.
  *
  * sbt forks a `runMain` from the base directory of the project it belongs to, so the runs of
  * [[detrLShapeTrain]] sit under the detr project rather than next to this one's.
  */
val DetectorCheckpointRoot: Option[String] = Some(s"../detr/${detr.lshape.CheckpointRoot}")

/** Axis of a batch of drawings. Named for the drawings rather than the batch because the graph
  * axes are already called after the boxes they run over.
  */
private trait Drawing derives Label

/** Axis of a model's parameters, flattened into one vector so that they can be counted. */
private trait Parameter derives Label

case class TrainState(
    params: EGTR.Params[Float32],
    optimizerState: LearningRateSchedulerState[EGTR.Params[Float32], AdamState],
    lastCost: Tensor0[Float32]
)

/** Trains a scene graph model on the l-shape drawings: `sbt "egtr/runMain egtr.lshape.egtrLShapeTrain"`.
  *
  * With no argument the detector is started from the newest run under
  * [[DetectorCheckpointRoot]], and from scratch if there is none; pass a run directory to pick
  * one. Starting from a trained detector is what the paper does — the relations are read out of
  * the detector's own attention, so they have little to say until the detection is roughly right,
  * and the [[EGTRLoss]] smoothing keeps them quiet until it is. The detector is not frozen: it
  * keeps training on the joint loss.
  *
  * The rectilinear run is [[egtrRectilinear6to18Train]], a copy of this one differing in the
  * values at the top. Two files rather than one parameterised by a corpus, so that a run can be
  * changed without touching the other.
  */
@main
def egtrLShapeTrain(detectorRun: String*): Unit =
  dimwit.initialize()

  val corpus = Corpus.LShape

  /** How many objects the decoder may answer with. As in [[detrLShapeTrain]]. */
  val numQueries = 32
  require(numQueries > corpus.maxNodes, s"$numQueries queries cannot answer for a drawing of up to ${corpus.maxNodes} objects")

  val numIterations = 150_000
  val batchSize = 64
  val learningRate = 3e-4f
  val weightDecay = 1e-4f
  val maxGradientNorm = 1.0f

  /** How long the rate climbs before it starts to fall. */
  val warmupSteps = 2_000

  /** Where the cosine bottoms out. Aligned with the other two models. */
  val finalLearningRate = 1e-4f

  val data = DrawingDataset.open(corpus)(Axis[Width], Axis[Height], Axis[Channel], Axis[BoundingBox], Axis[Relationship])(Split.Train)
  val batches = data.objectBatches(Axis[Drawing] -> batchSize)

  /** Linear warmup into a cosine decay to nothing — see [[detrLShapeTrain]] for why the rate has
    * to shrink. Held at a constant 3e-4 this model's detector moved 8% of its weight norm every
    * two thousand steps and its accuracy swung by five points; decayed, it can settle.
    */
  val schedule: LearningRateSchedule =
    LinearWarmup(Tensor0(learningRate), Tensor0(warmupSteps))
      .followBy(CosineDecay(Tensor0(learningRate), Tensor0(finalLearningRate), Tensor0(numIterations - warmupSteps)))
  val optimizer = LearningRateScheduler(lr => AdamW(Adam(learningRate = lr), Tensor0(weightDecay)), schedule)

  val detector = detectorRun.headOption
    .map(TensorTreeCheckpointer(_))
    .orElse(DetectorCheckpointRoot.flatMap(TensorTreeCheckpointer.latestIn(_))) match
    case Some(checkpoints) =>
      println(s"starting from the detector of ${checkpoints.rootPath}")
      checkpoints.loadLatest[DetectorState].getOrElse(sys.error(s"no checkpoint in ${checkpoints.rootPath}")).params
    case None =>
      println("starting the detector from scratch")
      DETR.Params.init(
        numLayers = 3,
        numHeads = 4,
        embedding = 128,
        numQueries = numQueries,
        patchSize = 16,
        key = Random.Key(0)
      )

  val initialParams = EGTR.Params.init(
    detector = detector,
    // One source projects a query into this, and a pair of queries into twice it, which is what
    // the heads read. The paper keeps both at the detector's embedding width.
    sourceExtent = 128,
    hiddenExtent = 128,
    key = Random.Key(1)
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
      state: TrainState
  ) =
    val (lastCost, gradients) = Autodiff.valueAndGrad(cost(images, objects, relations))(state.params)
    val clipped = gradients.clipGlobalNorm(Tensor0(maxGradientNorm))
    val (params, optimizerState) = optimizer.update(clipped, state.params, state.optimizerState)
    val newState = TrainState(params, optimizerState, lastCost)
    // The donated state and the batch are dead the moment the step returns, so their device
    // buffers go with them — as in detrLShapeTrain.
    summon[TensorTree[TrainState]].map(
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
        jitGradientStep(batch.images, batch.target.detection, batch.target.relations, state)
    .tapEvery(10):
      case (state, step) => println(monitor.report(step, state))
    .tapEvery(10_000):
      case (state, step) =>
        checkpointer.save(state, step)
        println(s"Step $step | checkpoint saved to ${checkpointer.rootPath}")
    .drop(numIterations)
    .next()
