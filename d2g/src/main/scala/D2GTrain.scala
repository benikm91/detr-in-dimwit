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
  * a change to how training works cannot reach one corpus and miss another — which is the failure
  * this codebase has already had once, in evaluation, and the reason scoring was pulled into a
  * single reporter.
  */
def d2gTrain(setup: D2GSetup): Unit =
  dimwit.initialize()
  println(s"training $setup")

  val nodes = Axis[Node] -> setup.nodeSlots
  val edges = Axis[Edge] -> setup.edgeSlots
  val data = DrawingDataset.open(setup.corpus)(Axis[Width], Axis[Height], Axis[Channel], Axis[Node], Axis[Edge])(Split.Train)
  val batches = data.batches(Axis[Batch] -> setup.batchSize)

  val (initKey, dataKey) = Random.Key(setup.seed).splitToTuple(2)

  /** Linear warmup into a cosine decay to a floor.
    *
    * Clipping cannot set the step size here — Adam normalises per parameter, so a clipped
    * gradient and an unclipped one move the weights by the same 0.2% of their norm — which leaves
    * the rate as the only thing that can. Held constant it never shrinks, so the model keeps
    * taking full sized steps long after it has found a solution and can only orbit one; decayed
    * it can actually settle on it.
    */
  val schedule: LearningRateSchedule =
    LinearWarmup(Tensor0(setup.learningRate), Tensor0(setup.warmupSteps))
      .followBy(
        CosineDecay(
          Tensor0(setup.learningRate),
          Tensor0(setup.finalLearningRate),
          Tensor0(setup.numIterations - setup.warmupSteps)
        )
      )
  val optimizer = LearningRateScheduler(lr => AdamW(Adam(learningRate = lr), Tensor0(setup.weightDecay)), schedule)

  val initialParams = D2G.Params.init(
    numLayers = setup.numLayers,
    numHeads = setup.numHeads,
    embedding = setup.embedding,
    nodes = setup.nodeSlots,
    edges = setup.edgeSlots,
    patchSize = setup.patchSize,
    canvas = Canvas,
    key = initKey
  )

  val (flattenParams, _) = TensorTree.ravel(initialParams, Axis[Parameter])
  println(s"parameters: ${flattenParams(initialParams).shape(Axis[Parameter])}")

  val nodeLoss = RemainingNodeLoss(VType[Float32], Canvas, setup.withPassThrough, setup.separation)
  val edgeLoss = RemainingEdgeLoss(VType[Float32], setup.withPassThrough, setup.separation)

  def cost(
      images: Tensor4[Batch, Width, Height, Channel, Float32],
      records: RecordBatch[Batch, Node, Edge],
      key: Key
  )(params: D2G.Params[Float32]): Tensor0[Float32] =
    val model = D2G(params, setup.predictionNoise)
    val keys = key.splitToTensor(images.shape.extent(Axis[Batch]))
    zipvmap(Axis[Batch])(images, records.nodeClass, records.startX, records.startY, records.endX, records.endY, records.edgeClass, records.subject, records.obj, keys):
      case (image, nodeClass, startX, startY, endX, endY, edgeClass, subject, obj, key) =>
        val target = Record(nodeClass, startX, startY, endX, endY, edgeClass, subject, obj)
        val scored = model(image, target, key.item)
        nodeLoss(scored.nodes, target.nodes).cost + edgeLoss(scored.edges, target.edges).cost
    .mean

  /** How often the two prediction tokens of a slot answered with the same node, over one batch
    * held aside for the purpose.
    *
    * Nothing about accuracy — it says whether the noise is doing anything at all. Near 100% and the
    * two tokens have collapsed into one and none of this is working; falling toward nothing and
    * they are answering independently.
    */
  def collisions(
      images: Tensor4[Batch, Width, Height, Channel, Float32],
      records: RecordBatch[Batch, Node, Edge],
      key: Key,
      params: D2G.Params[Float32]
  ): Tensor0[Float32] =
    val model = D2G(params, setup.predictionNoise)
    val keys = key.splitToTensor(images.shape.extent(Axis[Batch]))
    val counted = zipvmap(Axis[Batch])(images, records.nodeClass, records.startX, records.startY, records.endX, records.endY, records.edgeClass, records.subject, records.obj, keys):
      case (image, nodeClass, startX, startY, endX, endY, edgeClass, subject, obj, key) =>
        val target = Record(nodeClass, startX, startY, endX, endY, edgeClass, subject, obj)
        val scored = model(image, target, key.item)
        val (nodes, edges) = (nodeLoss(scored.nodes, target.nodes), edgeLoss(scored.edges, target.edges))
        (nodes.collided + edges.collided, nodes.asked + edges.asked)
    counted._1.sum / counted._2.sum

  def gradientStep(
      images: Tensor4[Batch, Width, Height, Channel, Float32],
      records: RecordBatch[Batch, Node, Edge],
      state: D2GTrainState
  ) =
    val (nextLinearization, forThisStep, forNoise) = state.linearization.splitToTuple(3)
    val (lastCost, gradients) = Autodiff.valueAndGrad(cost(images, records.permuted(forThisStep, nodes, edges), forNoise))(state.params)
    val (params, optimizerState) = optimizer.update(gradients.clipGlobalNorm(setup.maxGradientNorm), state.params, state.optimizerState)
    D2GTrainState(params, optimizerState, nextLinearization, lastCost)
  val jitGradientStep = jitDonatingUnsafe(gradientStep)

  /** One batch kept aside, so that the collision rate is read from the same drawings every time
    * and its movement is the model's rather than the batch's.
    */
  val watched = data.batches(Axis[Batch] -> setup.batchSize).next()
  val watchCollisions = jit: (params: D2G.Params[Float32], key: Key) =>
    collisions(watched.images, watched.target.permuted(key, nodes, edges), key, params)

  val checkpointer = TensorTreeCheckpointer.newIn(setup.checkpointRoot)
  val monitor = Monitor.ConcatMonitor[D2GTrainState](List(
    Monitor.StepMonitor(),
    Monitor.LossMonitor(_.lastCost.item),
    Monitor.LearningRateMonitor(schedule),
    Monitor.PerformanceMonitor(setup.batchSize)
  ))
  batches
    .scanLeft(D2GTrainState(initialParams, optimizer.init(initialParams), dataKey, Tensor0(-1f))):
      case (state, batch) => jitGradientStep(batch.images, batch.target, state)
    .tapEvery(100):
      case (state, step) =>
        val collided = watchCollisions(state.params, Random.Key(step)).item
        println(f"${monitor.report(step, state)} | Collisions: ${100f * collided}%5.1f%%")
    .tapEvery(setup.checkpointEvery):
      case (state, step) =>
        checkpointer.save(state, step)
        println(s"Step $step | checkpoint saved to ${checkpointer.rootPath}")
    .drop(setup.numIterations)
    .next()
