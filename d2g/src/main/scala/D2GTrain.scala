import dataset.Canvas
import dataset.Corpus
import dataset.DrawingDataset
import dataset.DrawingDataset.Split
import dataset.Record
import dataset.RecordBatch
import deepwit.checkpointing.TensorTreeCheckpointer
import deepwit.optimizer.clipGlobalNorm
import deepwit.training.Monitor
import deepwit.training.tapEvery
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

/** Axis of a batch of drawings. */
private trait Batch derives Label

/** Axis of a model's parameters, flattened into one vector so that they can be counted. */
private trait Parameter derives Label

case class D2GTrainState(
    params: D2G.Params[Float32],
    optimizerState: LearningRateSchedulerState[D2G.Params[Float32], AdamState],
    linearization: Key,
    lastCost: Tensor0[Float32]
)

/** Trains a transcription model on the corpus its [[D2GSetup]] names.
  *
  * Every step draws a fresh linearization of every drawing's record, so the same drawing is seen
  * with its nodes in a different order each time it comes round — which is the point: an order
  * the loss does not commit to is an order the model cannot learn to rely on.
  *
  * One implementation serves every corpus. What differs between them is held in the setup, so that
  * a change to how training works cannot reach one corpus and miss another.
  */
def d2gTrain(setup: D2GSetup): Unit =
  dimwit.initialize()
  println(s"training $setup")

  val nodes = Axis[Node] -> setup.nodeSlots
  val edges = Axis[Edge] -> setup.edgeSlots
  val data = DrawingDataset.open(setup.corpus)(Axis[Width], Axis[Height], Axis[Channel], Axis[Node], Axis[Edge])(Split.Train)
  val batches = data.batches(Axis[Batch] -> setup.batchSize)

  val (initKey, dataKey) = Random.Key(setup.seed).splitToTuple(2)

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

  val initialParams = D2G.Params.init(
    numLayers = setup.numLayers,
    numHeads = setup.numHeads,
    embedding = setup.embedding,
    nodes = setup.nodeSlots,
    edges = setup.edgeSlots,
    queries = setup.queryPool,
    patchSize = setup.patchSize,
    canvas = Canvas,
    key = initKey
  )

  val (flattenParams, _) = TensorTree.ravel(initialParams, Axis[Parameter])
  println(s"parameters: ${flattenParams(initialParams).shape(Axis[Parameter])}")

  val nodeLoss = RemainingNodeLoss(VType[Float32], Canvas)
  val edgeLoss = RemainingEdgeLoss(VType[Float32])

  def cost(
      images: Tensor4[Batch, Width, Height, Channel, Float32],
      records: RecordBatch[Batch, Node, Edge],
      asked: Key
  )(params: D2G.Params[Float32]): Tensor0[Float32] =
    val model = D2G(params)
    zipvmap(Axis[Batch])(images, records.nodeClass, records.startX, records.startY, records.endX, records.endY, records.edgeClass, records.subject, records.obj):
      case (image, nodeClass, startX, startY, endX, endY, edgeClass, subject, obj) =>
        val target = Record(nodeClass, startX, startY, endX, endY, edgeClass, subject, obj)
        val scored = model.logits(image, target, asked)
        nodeLoss(scored.nodes, target.nodes) + edgeLoss(scored.edges, target.edges)
    .mean

  def gradientStep(
      images: Tensor4[Batch, Width, Height, Channel, Float32],
      records: RecordBatch[Batch, Node, Edge],
      state: D2GTrainState
  ) =
    val (nextLinearization, forThisStep, forQueries) = state.linearization.splitToTuple(3)
    val (lastCost, gradients) = Autodiff.valueAndGrad(cost(images, records.permuted(forThisStep, nodes, edges), forQueries))(state.params)
    val (params, optimizerState) = optimizer.update(gradients.clipGlobalNorm(setup.maxGradientNorm), state.params, state.optimizerState)
    D2GTrainState(params, optimizerState, nextLinearization, lastCost)
  val jitGradientStep = jitDonatingUnsafe(gradientStep)

  val checkpointer = TensorTreeCheckpointer.newIn(setup.checkpointRoot)
  val monitor = Monitor.ConcatMonitor[D2GTrainState](List(
    Monitor.StepMonitor(),
    Monitor.LossMonitor(_.lastCost.item),
    Monitor.LearningRateMonitor(schedule),
    Monitor.PerformanceMonitor(setup.batchSize)
  ))
  batches
    .scanLeft(D2GTrainState(initialParams, optimizer.init(initialParams), dataKey, -1f)):
      case (state, batch) => jitGradientStep(batch.images, batch.target, state)
    .tapEvery(100):
      case (state, step) => println(monitor.report(step, state))
    .tapEvery(setup.checkpointEvery):
      case (state, step) =>
        checkpointer.save(state, step)
        println(s"Step $step | checkpoint saved to ${checkpointer.rootPath}")
    .drop(setup.numIterations)
    .next()
