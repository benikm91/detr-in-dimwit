import dataset.Detection
import dataset.DetectionBatch
import dataset.LShapeBatch
import dataset.LShapeDetectionDataset
import dataset.LShapeDetectionDataset.Split
import deepwit.Monitor
import deepwit.checkpointing.TensorTreeCheckpointer
import deepwit.tapEvery
import deepwit.transformer.MLPEmbeddingMixer
import deepwit.transformer.attention.Head
import deepwit.transformer.attention.HeadKey
import deepwit.transformer.attention.HeadQuery
import deepwit.transformer.attention.HeadValue
import dimwit.*
import dimwit.optimizer.Adam
import dimwit.optimizer.AdamState
import dimwit.tensor.Tensor4

/** Where [[detrTrain]] writes and [[detrEval]] reads its checkpoints. */
val CheckpointRoot = "out/detr"

private trait Batch derives Label

case class TrainState(
    params: DETR.Params[Float32],
    optimizerState: AdamState[DETR.Params[Float32]],
    lastCost: Tensor0[Float32]
)

@main
def detrTrain(): Unit =
  dimwit.initialize()

  val numIterations = 1_000
  val batchSize = 8
  val learningRate = 1e-4f

  val data = LShapeDetectionDataset.open(Axis[Width], Axis[Height], Axis[Channel], Axis[BoundingBox])(Split.Train)
  val batches = data.batches(Axis[Batch] -> batchSize, shuffle = Some(scala.util.Random(42)))

  val loss = HungarianLoss[Float32]()
  val optimizer = Adam(learningRate = Tensor0(learningRate))

  val initialParams = DETR.Params.init[Float32](numEncoderLayers = 3, numDecoderLayers = 3)(
    patchWidthExtent = Axis[Width] -> 16,
    patchHeightExtent = Axis[Height] -> 16,
    channelExtent = Axis[Channel] -> 1,
    boundingBoxExtent = Axis[BoundingBox] -> data.numQueries,
    headExtent = Axis[Head] -> 4,
    headQueryExtent = Axis[HeadQuery] -> 32,
    headKeyExtent = Axis[HeadKey] -> 32,
    headValueExtent = Axis[HeadValue] -> 32,
    embeddingExtent = Axis[DETR.Embedding] -> 128,
    embeddingMixedExtent = Axis[MLPEmbeddingMixer.EmbeddingMixed] -> 256,
    boxHiddenExtent = Axis[DETR.BoxHidden] -> 128,
    vtype = VType[Float32],
    key = Random.Key(0)
  )

  val predict = jit: (params: DETR.Params[Float32], images: Tensor4[Batch, Width, Height, Channel, Float32]) =>
    images.vmap(Axis[Batch])(DETR(params)(_).toTuple)

  /** Matching cannot be traced, so it runs on its own forward pass and the assignment it
    * decides on enters the gradient step as a constant.
    */
  def matchTargets(
      batch: LShapeBatch[Batch, Width, Height, Channel, BoundingBox],
      params: DETR.Params[Float32]
  ): DetectionBatch[Batch, BoundingBox] =
    val predictions = DETR.PredictionBatch(predict(params, batch.images))
    DetectionBatch.stacked:
      (0 until batch.images.shape(Axis[Batch])).map: sample =>
        loss.matchTargets(predictions.at(sample), batch.objects.at(sample))

  def cost(
      images: Tensor4[Batch, Width, Height, Channel, Float32],
      matched: DetectionBatch[Batch, BoundingBox]
  )(params: DETR.Params[Float32]): Tensor0[Float32] =
    val model = DETR(params)
    zipvmap(Axis[Batch])(images, matched.centerX, matched.centerY, matched.width, matched.height, matched.label):
      case (image, centerX, centerY, width, height, label) =>
        loss(model(image), Detection(centerX, centerY, width, height, label))
    .mean

  def optimize(
      images: Tensor4[Batch, Width, Height, Channel, Float32],
      matched: DetectionBatch[Batch, BoundingBox],
      state: TrainState
  ): TrainState =
    val (cost0, gradients) = Autodiff.valueAndGrad(cost(images, matched))(state.params)
    val (params, optimizerState) = optimizer.update(gradients, state.params, state.optimizerState)
    TrainState(params, optimizerState, cost0)
  val jitOptimize = jitDonatingUnsafe(optimize)

  def gradientStep(batch: LShapeBatch[Batch, Width, Height, Channel, BoundingBox], state: TrainState): TrainState =
    jitOptimize(batch.images, matchTargets(batch, state.params), state)

  val startedAt = java.time.LocalDateTime.now().format(java.time.format.DateTimeFormatter.ofPattern("yyyyMMdd_HHmmss"))
  val checkpointer = TensorTreeCheckpointer(s"$CheckpointRoot/$startedAt")
  val monitor = Monitor.default[TrainState](batchSize = batchSize, lossLens = _.lastCost.item)
  batches
    .scanLeft(TrainState(initialParams, optimizer.init(initialParams), Tensor0(-1f))):
      case (state, batch) =>
        dimwit.gc()
        gradientStep(batch, state)
    .tapEvery(10):
      case (state, step) => println(monitor.report(step, state))
    .tapEvery(250):
      case (state, step) =>
        checkpointer.save(state, step)
        println(s"Step $step | checkpoint saved to $CheckpointRoot/$startedAt")
    .drop(numIterations)
    .next()
