package d2g.floorplan.train

import d2g.*
import d2g.floorplan.model.*
import d2g.floorplan.config.*
import d2s.floorplan.train.D2STrainState
import dataset.DrawingDataset.Split
import dataset.drawingsOf
import dataset.History
import deepwit.checkpointing.TensorTreeCheckpointer
import deepwit.optimizer.LossScale
import deepwit.optimizer.allFinite
import deepwit.optimizer.clipGlobalNorm
import deepwit.optimizer.select
import deepwit.training.Monitor
import deepwit.training.tapEvery
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
import syntheticFloorplansDataset.FloorplanEdges
import syntheticFloorplansDataset.FloorplanNodes
import syntheticFloorplansDataset.FloorplanRecord
import syntheticFloorplansDataset.FloorplanRecordBatch
import syntheticFloorplansDataset.SyntheticFloorplans

import scala.language.implicitConversions

case class D2GTrainState(
    params: D2G.Params[Float32],
    optimizerState: LearningRateSchedulerState[D2G.Params.EdgeParams[Float32], AdamState],
    lossScale: LossScale,
    skippedSteps: Tensor0[Int32],
    linearization: Key,
    loss: Tensor0[Float32]
)

/** Trains a graph transcription model on the synthetic floor plans: the relationships, on top of
  * the nodes of a trained [[d2s.floorplan.model.D2S]] that does not move.
  *
  * Every step draws a fresh linearization of every drawing's record, so the same drawing is seen
  * with its nodes in a different order each time it comes round — which is the point: an order
  * the loss does not commit to is an order the model cannot learn to rely on.
  *
  * The model runs in Float16 against Float32 master weights, the loss in Float32. A [[LossScale]]
  * keeps the small gradients from rounding to zero, and a step whose gradients overflow is skipped.
  *
  * The batch is split over the devices: each takes the same step on its share of the drawings, and
  * the gradients are summed before the parameters move. A run on more GPUs therefore holds less per
  * device and takes the same steps — how many GPUs a run gets does not change what it learns.
  */
def trainTranscriber(setup: D2GSetup): Unit =
  println(s"training $setup")

  dimwit.initialize()
  trait Batch derives Label
  trait X derives MeshLabel
  val mesh = Mesh1(MeshAxis[X] -> Jax.devices.size)
  val batchSize = setup.batchSize
  require(batchSize % mesh.sizeOf(MeshAxis[X]) == 0, s"a batch of $batchSize does not split over $mesh")
  val numTotalSteps = setup.numSamples / batchSize
  val warmupSteps = setup.warmupSamples / batchSize
  val cooldownSteps = setup.cooldownSamples / batchSize
  val checkpointEvery = setup.checkpointEverySamples / batchSize
  require(numTotalSteps % checkpointEvery == 0, s"$numTotalSteps steps do not end on a checkpoint, which is where the cooldown ends")
  println(s"$mesh on ${Jax.devices.head.platform}, $batchSize drawings per step, $numTotalSteps steps")

  val nodes = Axis[Node] -> setup.nodeSlots
  val edges = Axis[Edge] -> setup.edgeSlots
  val data = SyntheticFloorplans.open(Axis[Width], Axis[Height], Axis[Channel], Axis[Node], Axis[Edge])(Split.Train, setup.canvas)

  val (initKey, dataKey) = Random.Key(setup.seed).splitToTuple(2)

  val schedule: LearningRateSchedule =
    LinearWarmup(setup.learningRate, warmupSteps)
      .followBy(ConstantLearningRate(setup.learningRate, numTotalSteps - warmupSteps - cooldownSteps))
      .followBy(CosineDecay(setup.learningRate, setup.finalLearningRate, cooldownSteps))
  val optimizer = LearningRateScheduler(lr => AdamW(Adam(learningRate = lr, beta2 = setup.adamBeta2), setup.weightDecay), schedule)

  val nodesFrom = TensorTreeCheckpointer.latestIn(setup.nodesFrom).getOrElse(sys.error(s"no run of d2s in ${setup.nodesFrom}"))
  val nodeWriter = nodesFrom.loadLatest[D2STrainState].getOrElse(sys.error(s"no checkpoint of d2s in ${nodesFrom.rootPath}")).params
  println(s"nodes written by ${nodesFrom.rootPath} at step ${nodesFrom.iterations.max}")

  val initialParams = D2G.Params.init(
    numLayers = setup.numLayers,
    numHeads = setup.numHeads,
    embedding = setup.embedding,
    nodes = setup.nodeSlots,
    edges = setup.edgeSlots,
    queries = setup.queryPool,
    canvas = setup.canvas,
    key = initKey
  ).copy(set = nodeWriter)

  trait Parameter derives Label
  val (flattenParams, _) = TensorTree.ravel(initialParams.edges, Axis[Parameter])
  println(s"parameters trained: ${flattenParams(initialParams.edges).shape(Axis[Parameter])}")

  val edgeLoss = RemainingEdgeLoss(VType[Float32])

  /** The mean loss over the drawings of a batch, whatever axis `S` they lie along. */
  def cost[S: Label](
      images: Tensor4[S, Width, Height, Channel, Float32],
      records: FloorplanRecordBatch[S, Node, Edge],
      asked: Key
  )(params: D2G.Params[Float32]): Tensor0[Float32] =
    val model = D2G(params.asFloats(VType[Float16]))
    zipvmap(Axis[S])(images.asFloat(VType[Float16]), records.nodeClass, records.startX, records.startY, records.endX, records.endY, records.relationship, records.subject, records.obj):
      case (image, nodeClass, startX, startY, endX, endY, relationship, subject, obj) =>
        val target = FloorplanRecord(FloorplanNodes(nodeClass, startX, startY, endX, endY), FloorplanEdges(relationship, subject, obj))
        val scored = model.logits(image, target, asked).asFloats(VType[Float32])
        edgeLoss(scored.edges, target.edges)
    .mean

  def gradientStep[S: Label](
      pixels: Tensor3[S, Height, Width, UInt8],
      records: FloorplanRecordBatch[S, Node, Edge],
      state: D2GTrainState
  ) =
    val images = drawingsOf(pixels, Axis[Channel])
    val (nextLinearization, forThisStep, forQueries) = state.linearization.splitToTuple(3)
    val lossScale = state.lossScale
    val (scaledCost, scaledGradients) = Autodiff.valueAndGrad(
      (edgeParams: D2G.Params.EdgeParams[Float32]) => lossScale.scaled(cost(images, records.permuted(forThisStep, nodes, edges, setup.nodeOrder.classRank), forQueries)(state.params.copy(edges = edgeParams)))
    )(state.params.edges)
    val gradients = lossScale.unscaled(scaledGradients)
    val (edgeParams, optimizerState) = optimizer.update(gradients.clipGlobalNorm(setup.maxGradientNorm), state.params.edges, state.optimizerState)
    val finite = allFinite(gradients)
    D2GTrainState(
      state.params.copy(edges = select(finite, edgeParams, state.params.edges)),
      select(finite, optimizerState, state.optimizerState),
      lossScale.next(finite),
      where(finite, state.skippedSteps, state.skippedSteps + 1),
      nextLinearization,
      where(finite, state.loss * 0.99f + scaledCost / lossScale.scale * 0.01f, state.loss)
    )
  val jitGradientStep = jitDonatingUnsafe(gradientStep[Batch |@| X])

  /** The batch sent from the host with every device holding its share of the drawings; the step
    * sees one axis.
    */
  def shard(
      batch: dataset.Batch[Batch, Width, Height, FloorplanRecordBatch[Batch, Node, Edge]]
  ): (Tensor3[Batch |@| X, Height, Width, UInt8], FloorplanRecordBatch[Batch |@| X, Node, Edge]) =
    val over = Axis[Batch] -> MeshAxis[X]
    val records = batch.target
    (
      batch.pixels.shard(mesh, over),
      FloorplanRecordBatch(
        nodeClass = records.nodeClass.shard(mesh, over),
        startX = records.startX.shard(mesh, over),
        startY = records.startY.shard(mesh, over),
        endX = records.endX.shard(mesh, over),
        endY = records.endY.shard(mesh, over),
        relationship = records.relationship.shard(mesh, over),
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
  val initialState = if taken == 0 then D2GTrainState(initialParams, optimizer.init(initialParams.edges), LossScale.initial(), 0, dataKey, 0f)
    else checkpointer.load[D2GTrainState](taken).getOrElse(sys.error(s"checkpoint $taken of ${checkpointer.rootPath} will not load"))
  if taken > 0 then println(s"continuing ${checkpointer.rootPath} from step $taken")
  val history = History(checkpointer.rootPath)
  /** Steps skipped all through a run mean the forward pass overflows, which no scale can fix. */
  val lossScaleMonitor = new Monitor[D2GTrainState]:
    def report(step: Int, state: D2GTrainState): String =
      f"Loss scale: 2^${math.log(state.lossScale.scale.item) / math.log(2)}%.0f, ${state.skippedSteps.item} steps skipped"
  val monitor = Monitor.ConcatMonitor[D2GTrainState](List(
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
