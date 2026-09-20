package d2g.train

import d2g.*
import d2g.model.*
import d2g.eval.*
import d2g.config.*
import dataset.Canvas
import dataset.Corpus
import dataset.DrawingDataset
import dataset.DrawingDataset.Split
import dataset.History
import dataset.Record
import dataset.RecordBatch
import deepwit.checkpointing.TensorTreeCheckpointer
import deepwit.optimizer.clipGlobalNorm
import deepwit.training.Monitor
import deepwit.training.tapEvery
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

case class D2GTrainState(
    params: D2G.Params[Float32],
    optimizerState: LearningRateSchedulerState[D2G.Params[Float32], AdamState],
    linearization: Key,
    loss: Tensor0[Float32]
)

/** Trains a transcription model on the corpus its [[D2GSetup]] names.
  *
  * Every step draws a fresh linearization of every drawing's record, so the same drawing is seen
  * with its nodes in a different order each time it comes round — which is the point: an order
  * the loss does not commit to is an order the model cannot learn to rely on.
  *
  * One implementation serves every corpus. What differs between them is held in the setup, so that
  * a change to how training works cannot reach one corpus and miss another.
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
  val numSteps = setup.numSamples / batchSize
  val warmupSteps = setup.warmupSamples / batchSize
  val cooldownSteps = setup.cooldownSamples / batchSize
  val checkpointEvery = setup.checkpointEverySamples / batchSize
  require(numSteps % checkpointEvery == 0, s"$numSteps steps do not end on a checkpoint, which is where the cooldown ends")
  println(s"$mesh on ${Jax.devices.head.platform}, $batchSize drawings per step, $numSteps steps")

  val nodes = Axis[Node] -> setup.nodeSlots
  val edges = Axis[Edge] -> setup.edgeSlots
  val data = DrawingDataset.open(setup.corpus)(Axis[Width], Axis[Height], Axis[Channel], Axis[Node], Axis[Edge])(Split.Train)
  val batches = data.batches(Axis[Batch] -> batchSize)

  val (initKey, dataKey) = Random.Key(setup.seed).splitToTuple(2)

  val schedule: LearningRateSchedule =
    LinearWarmup(setup.learningRate, warmupSteps)
      .followBy(ConstantLearningRate(setup.learningRate, numSteps - warmupSteps - cooldownSteps))
      .followBy(CosineDecay(setup.learningRate, setup.finalLearningRate, cooldownSteps))
  val optimizer = LearningRateScheduler(lr => AdamW(Adam(learningRate = lr), setup.weightDecay), schedule)

  val initialParams = D2G.Params.init(
    numLayers = setup.numLayers,
    numHeads = setup.numHeads,
    embedding = setup.embedding,
    nodes = setup.nodeSlots,
    edges = setup.edgeSlots,
    queries = setup.queryPool,
    canvas = Canvas,
    key = initKey
  )

  trait Parameter derives Label
  val (flattenParams, _) = TensorTree.ravel(initialParams, Axis[Parameter])
  println(s"parameters: ${flattenParams(initialParams).shape(Axis[Parameter])}")

  val nodeLoss = RemainingNodeLoss(VType[Float32], Canvas)
  val edgeLoss = RemainingEdgeLoss(VType[Float32])

  /** The mean loss over the drawings of a batch, whatever axis `S` they lie along. */
  def cost[S: Label](
      images: Tensor4[S, Width, Height, Channel, Float32],
      records: RecordBatch[S, Node, Edge],
      asked: Key
  )(params: D2G.Params[Float32]): Tensor0[Float32] =
    val model = D2G(params)
    zipvmap(Axis[S])(images, records.nodeClass, records.startX, records.startY, records.endX, records.endY, records.edgeClass, records.subject, records.obj):
      case (image, nodeClass, startX, startY, endX, endY, edgeClass, subject, obj) =>
        val target = Record(nodeClass, startX, startY, endX, endY, edgeClass, subject, obj)
        val scored = model.logits(image, target, asked)
        nodeLoss(scored.nodes, target.nodes) + edgeLoss(scored.edges, target.edges)
    .mean

  def gradientStep[S: Label](
      images: Tensor4[S, Width, Height, Channel, Float32],
      records: RecordBatch[S, Node, Edge],
      state: D2GTrainState
  ) =
    val (nextLinearization, forThisStep, forQueries) = state.linearization.splitToTuple(3)
    val (lastCost, gradients) = Autodiff.valueAndGrad(cost(images, records.permuted(forThisStep, nodes, edges), forQueries))(state.params)
    val (params, optimizerState) = optimizer.update(gradients.clipGlobalNorm(setup.maxGradientNorm), state.params, state.optimizerState)
    D2GTrainState(params, optimizerState, nextLinearization, state.loss * 0.99f + lastCost * 0.01f)
  val jitGradientStep = jitDonatingUnsafe(gradientStep[Batch |@| X])

  /** The batch with every device holding its share of the drawings; the step sees one axis. */
  def shard(
      batch: dataset.Batch[Batch, Width, Height, Channel, RecordBatch[Batch, Node, Edge]]
  ): (Tensor4[Batch |@| X, Width, Height, Channel, Float32], RecordBatch[Batch |@| X, Node, Edge]) =
    val over = Axis[Batch] -> MeshAxis[X]
    val records = batch.target
    (
      batch.images.shard(mesh, over),
      RecordBatch(
        nodeClass = records.nodeClass.shard(mesh, over),
        startX = records.startX.shard(mesh, over),
        startY = records.startY.shard(mesh, over),
        endX = records.endX.shard(mesh, over),
        endY = records.endY.shard(mesh, over),
        edgeClass = records.edgeClass.shard(mesh, over),
        subject = records.subject.shard(mesh, over),
        obj = records.obj.shard(mesh, over)
      )
    )

  val checkpointer = TensorTreeCheckpointer.newIn(setup.checkpointRoot)
  val history = History()
  val monitor = Monitor.ConcatMonitor[D2GTrainState](List(
    Monitor.StepMonitor(),
    Monitor.LossMonitor(_.loss.item),
    Monitor.LearningRateMonitor(schedule),
    Monitor.PerformanceMonitor(batchSize)
  ))

  val started = System.nanoTime
  batches
    .scanLeft(D2GTrainState(initialParams, optimizer.init(initialParams), dataKey, 0f)):
      case (state, batch) =>
        val (images, records) = shard(batch)
        jitGradientStep(images, records, state)
    .tapEvery(100):
      case (state, step) => println(monitor.report(step, state))
    .tapEvery(checkpointEvery):
      case (state, step) =>
        checkpointer.save(state, step)
        history.add(step, state.loss.item)
        println(s"Step $step | checkpoint saved to ${checkpointer.rootPath}")
    .drop(numSteps)
    .next()
  Runs.noteTrainingSeconds(checkpointer.rootPath, (System.nanoTime - started) / 1_000_000_000)
  Runs.noteTrainingSeconds(checkpointer.rootPath, (System.nanoTime - started) / 1_000_000_000)
