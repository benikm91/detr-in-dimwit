package detr.train

import detr.*
import detr.model.*
import detr.eval.*
import detr.config.*
import dataset.Corpus
import dataset.RecordBatch
import dataset.RecordNodes
import dataset.DrawingDataset
import dataset.History
import dataset.DrawingDataset.Split
import dataset.drawingsOf
import deepwit.checkpointing.TensorTreeCheckpointer
import deepwit.training.Monitor
import deepwit.training.tapEvery
import deepwit.attention.Head
import deepwit.attention.HeadKey
import deepwit.attention.HeadQuery
import deepwit.attention.HeadValue
import deepwit.optimizer.LossScale
import deepwit.optimizer.allFinite
import deepwit.optimizer.clipGlobalNorm
import deepwit.optimizer.select
import dataset.Runs
import dimwit.*
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
import dimwit.TreeOf.ops.asFloats

import scala.language.implicitConversions

private trait Batch derives Label

/** Mesh axis the batch is split over. */
private trait X derives MeshLabel

/** Axis of a model's parameters, flattened into one vector so that they can be counted. */
trait Parameter derives Label

case class TrainState(
    params: DETR.Params[Float32],
    optimizerState: LearningRateSchedulerState[DETR.Params[Float32], AdamState],
    lossScale: LossScale,
    skippedSteps: Tensor0[Int32],
    loss: Tensor0[Float32]
)

/** Trains a detector on the corpus its [[DETRSetup]] names.
  *
  * The model runs in Float16 against Float32 master weights, the loss and its matching in
  * Float32. A [[LossScale]] keeps the small gradients from rounding to zero, and a step whose
  * gradients overflow is skipped.
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

  val data = DrawingDataset.open(setup.corpus)(Axis[Width], Axis[Height], Axis[Channel], Axis[Node], Axis[Relationship])(Split.Train)

  /** Warmup, then the rate held, then a cosine cooldown over the last stretch. Adam orbits a
    * solution at a distance the rate sets, which is what the cooldown closes; holding the rate
    * until then leaves a checkpoint comparable with one from a run of another length.
    */
  val schedule: LearningRateSchedule =
    LinearWarmup(setup.learningRate, warmupSteps)
      .followBy(ConstantLearningRate(setup.learningRate, numTotalSteps - warmupSteps - cooldownSteps))
      .followBy(CosineDecay(setup.learningRate, setup.finalLearningRate, cooldownSteps))
  val optimizer = LearningRateScheduler(lr => AdamW(Adam(learningRate = lr, beta2 = setup.adamBeta2), setup.weightDecay), schedule)

  val initialParams = DETR.Params.init(
    numLayers = setup.numLayers,
    numHeads = setup.numHeads,
    embedding = setup.embedding,
    numQueries = setup.numQueries,
    canvas = setup.corpus.canvas,
    key = Random.Key(setup.seed)
  )

  val (flattenParams, _) = TensorTree.ravel(initialParams, Axis[Parameter])
  println(s"parameters: ${flattenParams(initialParams).shape(Axis[Parameter])}")

  val loss = HungarianLoss(VType[Float32], setup.corpus.canvas)

  def cost(
      imgs: Tensor4[Batch |@| X, Width, Height, Channel, Float32],
      records: RecordBatch[Batch |@| X, Node, Relationship]
  )(params: DETR.Params[Float32]): Tensor0[Float32] =
    val model = DETR(params.asFloats(VType[Float16]))
    zipvmap(Axis[Batch |@| X])(imgs.asFloat(VType[Float16]), records.nodeClass, records.construction, records.startX, records.startY, records.endX, records.endY, records.midX, records.midY):
      case (img, nodeClass, construction, startX, startY, endX, endY, midX, midY) =>
        loss(model.logits(img).asFloats(VType[Float32]), RecordNodes(nodeClass, construction, startX, startY, endX, endY, midX, midY))
    .mean

  def gradientStep(
      pixels: Tensor3[Batch |@| X, Height, Width, UInt8],
      records: RecordBatch[Batch |@| X, Node, Relationship],
      state: TrainState
  ) =
    val lossScale = state.lossScale
    val (scaledCost, scaledGradients) = Autodiff.valueAndGrad(
      (params: DETR.Params[Float32]) => lossScale.scaled(cost(drawingsOf(pixels, Axis[Channel]), records)(params))
    )(state.params)
    val gradients = lossScale.unscaled(scaledGradients)
    val (params, optimizerState) = optimizer.update(gradients.clipGlobalNorm(setup.maxGradientNorm), state.params, state.optimizerState)
    val finite = allFinite(gradients)
    TrainState(
      select(finite, params, state.params),
      select(finite, optimizerState, state.optimizerState),
      lossScale.next(finite),
      where(finite, state.skippedSteps, state.skippedSteps + 1),
      where(finite, state.loss * 0.99f + scaledCost / lossScale.scale * 0.01f, state.loss)
    )
  val jitGradientStep = jitDonatingUnsafe(gradientStep)

  /** The batch sent from the host with every device holding its share of the drawings; the step
    * sees one axis.
    */
  def shard(
      batch: dataset.Batch[Batch, Width, Height, RecordBatch[Batch, Node, Relationship]]
  ): (Tensor3[Batch |@| X, Height, Width, UInt8], RecordBatch[Batch |@| X, Node, Relationship]) =
    val over = Axis[Batch] -> MeshAxis[X]
    val records = batch.target
    (
      batch.pixels.shard(mesh, over),
      RecordBatch(
        nodeClass = records.nodeClass.shard(mesh, over),
        construction = records.construction.shard(mesh, over),
        startX = records.startX.shard(mesh, over),
        startY = records.startY.shard(mesh, over),
        endX = records.endX.shard(mesh, over),
        endY = records.endY.shard(mesh, over),
        midX = records.midX.shard(mesh, over),
        midY = records.midY.shard(mesh, over),
        edgeClass = records.edgeClass.shard(mesh, over),
        subject = records.subject.shard(mesh, over),
        obj = records.obj.shard(mesh, over)
      )
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
  val initialState = if taken == 0 then TrainState(initialParams, optimizer.init(initialParams), LossScale.initial(), 0, 0f)
    else checkpointer.load[TrainState](taken).getOrElse(sys.error(s"checkpoint $taken of ${checkpointer.rootPath} will not load"))
  if taken > 0 then println(s"continuing ${checkpointer.rootPath} from step $taken")
  val history = History(checkpointer.rootPath)
  /** Steps skipped all through a run mean the forward pass overflows, which no scale can fix. */
  val lossScaleMonitor = new Monitor[TrainState]:
    def report(step: Int, state: TrainState): String =
      f"Loss scale: 2^${math.log(state.lossScale.scale.item) / math.log(2)}%.0f, ${state.skippedSteps.item} steps skipped"
  val monitor = Monitor.ConcatMonitor[TrainState](List(
    Monitor.StepMonitor(),
    Monitor.LossMonitor(_.loss.item),
    Monitor.LearningRateMonitor(schedule),
    lossScaleMonitor,
    Monitor.PerformanceMonitor(batchSize)
  ))

  val started = System.nanoTime
  data.batches(Axis[Batch] -> batchSize, afterSteps = taken)
    .scanLeft(initialState):
      case (state, batch) =>
        val (images, records) = shard(batch)
        jitGradientStep(images, records, state)
    .tapEvery(100):
      case (state, step) => println(monitor.report(taken + step, state))
    .tapEvery(checkpointEvery):
      case (state, step) =>
        checkpointer.save(state, taken + step)
        history.add(taken + step, state.loss.item)
        println(s"Step ${taken + step} | checkpoint saved to ${checkpointer.rootPath}")
    .drop(numTotalSteps - taken)
    .next()
  Runs.noteTrainingSeconds(checkpointer.rootPath, (System.nanoTime - started) / 1_000_000_000)
