package d2tok.sketchgraph.train

import d2tok.*
import d2tok.model.*
import d2tok.sketch.train.D2TokTrainState
import d2tok.sketchgraph.config.*
import dataset.DrawingDataset
import dataset.DrawingDataset.Split
import dataset.History
import dataset.Record
import dataset.RecordBatch
import dataset.Runs
import dataset.drawingsOf
import deepwit.checkpointing.TensorTreeCheckpointer
import deepwit.optimizer.ConstantLearningRate
import deepwit.optimizer.CosineDecay
import deepwit.optimizer.LearningRateSchedule
import deepwit.optimizer.LearningRateScheduler
import deepwit.optimizer.LinearWarmup
import deepwit.optimizer.LossScale
import deepwit.optimizer.allFinite
import deepwit.optimizer.clipGlobalNorm
import deepwit.optimizer.select
import deepwit.training.Monitor
import deepwit.training.tapEvery
import dimwit.*
import dimwit.jax.Jax
import dimwit.optimizer.Adam
import dimwit.optimizer.AdamW
import dimwit.sharding.*
import dimwit.tensor.Tensor4

import scala.language.implicitConversions

/** Trains a language model of records with their relationships, every part from nothing: every word
  * of a record — its nodes, then its relationships — predicted from the document and the words
  * before it.
  *
  * Otherwise a step is the one of [[d2tok.sketch.train.trainWordTranscriber]], but the model runs
  * in Float32; its state is the same.
  */
def trainGraphTranscriber(setup: D2TokGraphSetup): Unit =
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

  val tokens = Axis[Token] -> setup.tokens
  val (nodes, relationships) = (Axis[Node] -> setup.corpus.maxNodes, Axis[Relationship] -> setup.corpus.maxEdges)
  val data = DrawingDataset.open(setup.corpus)(Axis[Width], Axis[Height], Axis[Channel], Axis[Node], Axis[Relationship])(Split.Train)

  val (initKey, dataKey) = Random.Key(setup.seed).splitToTuple(2)

  val schedule: LearningRateSchedule =
    LinearWarmup(setup.learningRate, warmupSteps)
      .followBy(ConstantLearningRate(setup.learningRate, numTotalSteps - warmupSteps - cooldownSteps))
      .followBy(CosineDecay(setup.learningRate, setup.finalLearningRate, cooldownSteps))
  val optimizer = LearningRateScheduler(lr => AdamW(Adam(learningRate = lr, beta2 = setup.adamBeta2), setup.weightDecay), schedule)

  val initialParams = D2Tok.Params.init(
    numLayers = setup.numLayers,
    numHeads = setup.numHeads,
    embedding = setup.embedding,
    words = setup.words.size,
    tokens = setup.tokens,
    key = initKey
  )

  trait Parameter derives Label
  val (flattenParams, _) = TensorTree.ravel(initialParams, Axis[Parameter])
  println(s"parameters: ${flattenParams(initialParams).shape(Axis[Parameter])}")

  val positions = Tensor1(tokens.axis).fromRange(0 until tokens.size)

  /** The mean loss over the drawings of a batch, whatever axis `S` they lie along: the cross
    * entropy of every word of a record, its end among them, against the words before it.
    */
  def cost[S: Label](
      images: Tensor4[S, Width, Height, Channel, Float32],
      records: RecordBatch[S, Node, Relationship]
  )(params: D2Tok.Params[Float32]): Tensor0[Float32] =
    val model = D2Tok(params)
    zipvmap(Axis[S])(images, records.nodeClass, records.construction, records.startX, records.startY, records.endX, records.endY, records.midX, records.midY, records.edgeClass, records.subject, records.obj):
      case (image, nodeClass, construction, startX, startY, endX, endY, midX, midY, edgeClass, subject, obj) =>
        val record = Record(nodeClass, construction, startX, startY, endX, endY, midX, midY, edgeClass, subject, obj)
        val (written, count) = setup.words.written(record, tokens)
        val read = concatenate(Tensor0(setup.words.start).prependAxis(Axis[Token]), written.slice(Axis[Token].at(0 until tokens.size - 1)), Axis[Token])
        val scored = model.logits(image, read)
        val missed = zipvmap(Axis[Token])(scored, written)((scores, word) => -scores.logSoftmax(Axis[Word]).slice(Axis[Word].at(word)))
        (missed * (positions <=! (count - 1)).asFloat(VType[Float32])).sum / count.asFloat(VType[Float32])
    .mean

  def gradientStep[S: Label](
      pixels: Tensor3[S, Height, Width, UInt8],
      records: RecordBatch[S, Node, Relationship],
      state: D2TokTrainState
  ) =
    val images = drawingsOf(pixels, Axis[Channel])
    val (nextLinearization, forThisStep) = state.linearization.splitToTuple(2)
    val laidOut = records.permuted(forThisStep, nodes, relationships, setup.nodeClassRank, setup.relationshipOrder.classRank)
    val lossScale = state.lossScale
    val (scaledCost, scaledGradients) = Autodiff.valueAndGrad(
      (params: D2Tok.Params[Float32]) => lossScale.scaled(cost(images, laidOut)(params))
    )(state.params)
    val gradients = lossScale.unscaled(scaledGradients)
    val (params, optimizerState) = optimizer.update(gradients.clipGlobalNorm(setup.maxGradientNorm), state.params, state.optimizerState)
    val finite = allFinite(gradients)
    D2TokTrainState(
      select(finite, params, state.params),
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

  /** The run's own folder, continued where it already holds checkpoints, as in
    * [[d2tok.sketch.train.trainWordTranscriber]].
    */
  val checkpointer = TensorTreeCheckpointer.latestIn(setup.checkpointRoot) match
    case Some(started) => TensorTreeCheckpointer(started.rootPath, overwrite = true)
    case None          => TensorTreeCheckpointer.newIn(setup.checkpointRoot)
  val taken = checkpointer.iterations.maxOption.getOrElse(0)
  val initialState = if taken == 0 then D2TokTrainState(initialParams, optimizer.init(initialParams), LossScale.initial(), 0, dataKey, 0f)
    else checkpointer.load[D2TokTrainState](taken).getOrElse(sys.error(s"checkpoint $taken of ${checkpointer.rootPath} will not load"))
  if taken > 0 then println(s"continuing ${checkpointer.rootPath} from step $taken")
  val history = History(checkpointer.rootPath)
  /** Steps skipped all through a run mean the forward pass overflows, which no scale can fix. */
  val lossScaleMonitor = new Monitor[D2TokTrainState]:
    def report(step: Int, state: D2TokTrainState): String =
      f"Loss scale: 2^${math.log(state.lossScale.scale.item) / math.log(2)}%.0f, ${state.skippedSteps.item} steps skipped"
  val monitor = Monitor.ConcatMonitor[D2TokTrainState](List(
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
