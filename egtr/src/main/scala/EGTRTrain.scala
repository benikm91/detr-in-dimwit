import dataset.Box
import dataset.Detection
import dataset.DetectionBatch
import dataset.LShapeDataset
import dataset.LShapeDataset.Split
import dataset.RelationClasses
import deepwit.checkpointing.TensorTreeCheckpointer
import deepwit.training.Monitor
import deepwit.training.tapEvery
import deepwit.optimizer.clipGlobalNorm
import dimwit.*
import dimwit.optimizer.Adam
import dimwit.optimizer.AdamState
import dimwit.optimizer.AdamW
import dimwit.tensor.Tensor4

/** Where [[egtrTrain]] writes and [[egtrEval]] reads its checkpoints. */
val EGTRCheckpointRoot = "out/egtr"

/** Where [[egtrTrain]] looks for a detector to start from.
  *
  * sbt forks a `runMain` from the base directory of the project it belongs to, so the runs of
  * [[detrTrain]] sit under the detr project rather than next to this one's.
  */
val DetectorCheckpointRoot = s"../detr/$CheckpointRoot"

/** Axis of a batch of drawings. Named for the drawings rather than the batch because the graph
  * axes are already called after the boxes they run over.
  */
private trait Drawing derives Label

case class EGTRTrainState(
    params: EGTR.Params[Float32],
    optimizerState: AdamState[EGTR.Params[Float32]],
    lastCost: Tensor0[Float32]
)

/** Trains a scene graph model: `sbt "egtr/runMain egtrTrain"`.
  *
  * With no argument the detector is started from the newest [[detrTrain]] run, and from scratch
  * if there is none; pass a run directory to pick one. Starting from a trained detector is what
  * the paper does — the relations are read out of the detector's own attention, so they have
  * little to say until the detection is roughly right, and the [[EGTRLoss]] smoothing keeps them
  * quiet until it is. The detector is not frozen: it keeps training on the joint loss.
  */
@main
def egtrTrain(detectorRun: String*): Unit =
  dimwit.initialize()

  val numIterations = 100_000
  // Half of what the detector trains with: a relation representation per pair of queries is a
  // far larger activation than anything in the detector, and it is kept per source.
  val batchSize = 32
  val learningRate = 3e-4f
  val weightDecay = 1e-4f
  val maxGradientNorm = 0.1f

  val data = LShapeDataset.open(Axis[Width], Axis[Height], Axis[Channel], Axis[BoundingBox])(Split.Train)
  val shuffle = scala.util.Random(42)
  val batches = Iterator.continually(data.objectBatches(Axis[Drawing] -> batchSize, shuffle = Some(shuffle))).flatten

  val optimizer = AdamW(
    Adam(learningRate = Tensor0(learningRate)),
    weightDecayFactor = Tensor0(weightDecay)
  )

  val detector = detectorRun.headOption
    .map(TensorTreeCheckpointer(_))
    .orElse(TensorTreeCheckpointer.latestIn(DetectorCheckpointRoot)) match
    case Some(checkpoints) =>
      println(s"starting from the detector of ${checkpoints.rootPath}")
      checkpoints.loadLatest[TrainState].getOrElse(sys.error(s"no checkpoint in ${checkpoints.rootPath}")).params
    case None =>
      println(s"no detector in $DetectorCheckpointRoot, starting from scratch")
      DETR.Params.init(
        numLayers = 3,
        numHeads = 4,
        embedding = 128,
        numQueries = 32,
        patchSize = 10,
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
    val clipped = gradients.clipGlobalNorm(Tensor0(maxGradientNorm))
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

  val checkpointer = TensorTreeCheckpointer.newIn(EGTRCheckpointRoot)
  val monitor = Monitor.default[EGTRTrainState](batchSize = batchSize, lossLens = _.lastCost.item)
  batches
    .scanLeft(EGTRTrainState(initialParams, optimizer.init(initialParams), Tensor0(-1f))):
      case (state, batch) =>
        dimwit.gc()
        jitGradientStep(batch.images, batch.target.detection, batch.target.relations, state)
    .tapEvery(10):
      case (state, step) => println(monitor.report(step, state))
    .tapEvery(1_000):
      case (state, step) =>
        checkpointer.save(state, step)
        println(s"Step $step | checkpoint saved to ${checkpointer.rootPath}")
    .drop(numIterations)
    .next()
