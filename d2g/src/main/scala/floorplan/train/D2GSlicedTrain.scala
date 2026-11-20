package d2g.floorplan.train

import d2g.*
import d2g.floorplan.model.*
import d2g.floorplan.config.*
import d2s.floorplan.train.RemainingNodeLoss
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
import dimwit.TreeOf.ops.++
import dimwit.jax.Jax
import dimwit.sharding.*
import deepwit.optimizer.ConstantLearningRate
import deepwit.optimizer.CosineDecay
import deepwit.optimizer.LearningRateSchedule
import deepwit.optimizer.LearningRateScheduler
import deepwit.optimizer.LinearWarmup
import dimwit.optimizer.Adam
import dimwit.optimizer.AdamW
import dimwit.tensor.Tensor4
import syntheticFloorplansDataset.FloorplanEdges
import syntheticFloorplansDataset.FloorplanNodes
import syntheticFloorplansDataset.FloorplanRecord
import syntheticFloorplansDataset.FloorplanRecordBatch
import syntheticFloorplansDataset.SyntheticFloorplans

import scala.language.implicitConversions

/** Trains every part of a graph transcription model from nothing, as [[trainWhole]] does, for a
  * batch too big for the GPUs: a step's batch is read in `slices` that are sent through the model
  * one after another, and the step is taken on the gradients summed over them.
  */
def trainWholeInSlices(setup: D2GSetup, slices: Int): Unit =
  println(s"training every part, from nothing, in $slices slices a step, $setup")

  dimwit.initialize()
  trait Batch derives Label
  trait X derives MeshLabel
  val mesh = Mesh1(MeshAxis[X] -> Jax.devices.size)
  val batchSize = setup.batchSize
  val sliceSize = batchSize / slices
  require(batchSize % slices == 0 && sliceSize % mesh.sizeOf(MeshAxis[X]) == 0, s"a batch of $batchSize does not cut into $slices slices that split over $mesh")
  val numTotalSteps = setup.numSamples / batchSize
  val warmupSteps = setup.warmupSamples / batchSize
  val cooldownSteps = setup.cooldownSamples / batchSize
  val checkpointEvery = setup.checkpointEverySamples / batchSize
  require(numTotalSteps % checkpointEvery == 0, s"$numTotalSteps steps do not end on a checkpoint, which is where the cooldown ends")
  println(s"$mesh on ${Jax.devices.head.platform}, $batchSize drawings per step in slices of $sliceSize, $numTotalSteps steps")

  val nodes = Axis[Node] -> setup.nodeSlots
  val edges = Axis[Edge] -> setup.edgeSlots
  val data = SyntheticFloorplans.open(Axis[Width], Axis[Height], Axis[Channel], Axis[Node], Axis[Edge])(setup.corpus, Split.Train, setup.canvas)

  val schedule: LearningRateSchedule =
    LinearWarmup(setup.learningRate, warmupSteps)
      .followBy(ConstantLearningRate(setup.learningRate, numTotalSteps - warmupSteps - cooldownSteps))
      .followBy(CosineDecay(setup.learningRate, setup.finalLearningRate, cooldownSteps))
  val optimizer = LearningRateScheduler(lr => AdamW(Adam(learningRate = lr, beta2 = setup.adamBeta2), setup.weightDecay), schedule)

  val (initKey, dataKey) = Random.Key(setup.seed).splitToTuple(2)

  val nodeLoss = RemainingNodeLoss(VType[Float32], setup.canvas, setup.nodeOrder)
  val edgeLoss = RemainingEdgeLoss(VType[Float32], setup.relationshipOrder)

  /** The mean loss over the drawings of a slice, whatever axis `S` they lie along. */
  def cost[S: Label](
      images: Tensor4[S, Width, Height, Channel, Float32],
      records: FloorplanRecordBatch[S, Node, Edge],
      asked: Key
  )(params: D2G.Params[Float32]): Tensor0[Float32] =
    val model = D2G(params)
    zipvmap(Axis[S])(images, records.nodeClass, records.startX, records.startY, records.endX, records.endY, records.relationship, records.subject, records.obj):
      case (image, nodeClass, startX, startY, endX, endY, relationship, subject, obj) =>
        val target = FloorplanRecord(FloorplanNodes(nodeClass, startX, startY, endX, endY), FloorplanEdges(relationship, subject, obj))
        val scored = model.logits(image, target, asked)
        nodeLoss(scored.nodes, target.nodes) + edgeLoss(scored.edges, target)
    .mean

  type Scaled = (Tensor0[Float32], Grad[D2G.Params[Float32]])

  /** A slice's share of the step's cost, and its gradients, both scaled by the step's loss scale:
    * summed over the slices, they are the batch's.
    */
  def sliceGradients[S: Label](
      pixels: Tensor3[S, Height, Width, UInt8],
      records: FloorplanRecordBatch[S, Node, Edge],
      linearization: Key,
      state: D2GWholeTrainState
  ): Scaled =
    val images = drawingsOf(pixels, Axis[Channel])
    val (forThisSlice, forQueries) = linearization.splitToTuple(2)
    Autodiff.valueAndGrad(
      (params: D2G.Params[Float32]) => state.lossScale.scaled(cost(images, records.permuted(forThisSlice, nodes, edges, setup.nodeOrder.classRank, setup.relationshipOrder.kindRank), forQueries)(params) / slices)
    )(state.params)
  val jitSliceGradients = jit(sliceGradients[Batch |@| X])
  val jitAdded = jit((a: Scaled, b: Scaled) => (a._1 + b._1, Grad(a._2.value ++ b._2.value)))

  /** The step of [[trainWhole]], on the cost and gradients of the whole batch. */
  def gradientStep(summed: Scaled, state: D2GWholeTrainState) =
    val (scaledCost, scaledGradients) = summed
    val lossScale = state.lossScale
    val gradients = lossScale.unscaled(scaledGradients)
    val (params, optimizerState) = optimizer.update(gradients.clipGlobalNorm(setup.maxGradientNorm), state.params, state.optimizerState)
    val finite = allFinite(gradients)
    D2GWholeTrainState(
      select(finite, params, state.params),
      select(finite, optimizerState, state.optimizerState),
      lossScale.next(finite),
      where(finite, state.skippedSteps, state.skippedSteps + 1),
      state.linearization,
      where(finite, state.loss * 0.99f + scaledCost / lossScale.scale * 0.01f, state.loss)
    )
  val jitGradientStep = jitDonatingUnsafe(gradientStep)

  /** The slice sent from the host with every device holding its share of the drawings. */
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

  /** The run's own folder, continued where it already holds checkpoints, as in [[trainWhole]]. */
  val checkpointer = TensorTreeCheckpointer.latestIn(setup.checkpointRoot) match
    case Some(started) => TensorTreeCheckpointer(started.rootPath, overwrite = true)
    case None          => TensorTreeCheckpointer.newIn(setup.checkpointRoot)
  val taken = checkpointer.iterations.maxOption.getOrElse(0)
  val initialState =
    if taken == 0 then
      val initialParams = D2G.Params.init(setup.numLayers, setup.numHeads, setup.embedding, setup.nodeSlots, setup.edgeSlots, setup.queryPool, setup.canvas, initKey)
      D2GWholeTrainState(initialParams, optimizer.init(initialParams), LossScale.initial(), 0, dataKey, 0f)
    else checkpointer.load[D2GWholeTrainState](taken).getOrElse(sys.error(s"checkpoint $taken of ${checkpointer.rootPath} will not load"))
  if taken > 0 then println(s"continuing ${checkpointer.rootPath} from step $taken")
  val history = History(checkpointer.rootPath)
  val lossScaleMonitor = new Monitor[D2GWholeTrainState]:
    def report(step: Int, state: D2GWholeTrainState): String =
      f"Loss scale: 2^${math.log(state.lossScale.scale.item) / math.log(2)}%.0f, ${state.skippedSteps.item} steps skipped"
  val monitor = Monitor.ConcatMonitor[D2GWholeTrainState](List(
    Monitor.StepMonitor(),
    Monitor.LossMonitor(_.loss.item),
    Monitor.LearningRateMonitor(schedule),
    lossScaleMonitor,
    Monitor.PerformanceMonitor(batchSize)
  ))

  val started = System.nanoTime
  data.batches(Axis[Batch] -> sliceSize, afterSteps = taken * slices)
    .grouped(slices)
    .scanLeft(initialState):
      case (state, batch) =>
        val (nextLinearization, forThisStep) = state.linearization.splitToTuple(2)
        val summed = batch.iterator.zip(forThisStep.split(slices))
          .map: (slice, linearization) =>
            val (images, records) = shard(slice)
            jitSliceGradients(images, records, linearization, state)
          .reduce(jitAdded)
        jitGradientStep(summed, state.copy(linearization = nextLinearization))
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
