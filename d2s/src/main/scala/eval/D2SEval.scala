package d2s.eval

import d2s.*
import d2s.model.*
import d2s.train.*
import d2s.config.*
import NodeHead.NodeLogits
import dataset.Canvas
import dataset.Corpus
import dataset.DrawingDataset
import dataset.DrawingDataset.Split
import dataset.Metrics
import dataset.NodeClass
import dataset.NodeClasses
import dataset.greyLevels
import dataset.RecordBatch
import dataset.RecordNodes
import dataset.RecordDrawing
import dataset.RecordGraph
import dataset.RecordScoring
import dataset.Runs
import dataset.Tolerances
import dataset.Transcripts
import dataset.at
import dataset.report
import deepwit.checkpointing.TensorTreeCheckpointer
import dimwit.*
import dimwit.tensor.Tensor4
import plotwit.*
import viz.PlotTargets.websocket

import scala.language.implicitConversions

/** Scores every checkpoint of the newest run of a setup on the whole validation split, the last
  * checkpoint first, and writes the metrics as one CSV, and what the last checkpoint wrote down
  * for every drawing as [[Transcripts]].
  */
def scoreSetTranscriber(setup: D2SSetup, size: String): Unit =
  dimwit.initialize()

  val checkpoints = TensorTreeCheckpointer.latestIn(setup.checkpointRoot).getOrElse(sys.error(s"no training run in ${setup.checkpointRoot}"))
  println(s"reading ${checkpoints.rootPath}")
  val nodes = Axis[Node] -> setup.nodeSlots
  val data = open(setup, Split.Validation)
  println(s"pool of ${setup.queryPool} queries, ${data.numSamples} drawings\n")

  val transcriber = Transcriber(nodes, TranscribedTogether)

  /** Every drawing of the split transcribed by a model, beside the nodes of the record it was
    * rendered from — a set holds no relationships, so the record it is held against holds none
    * either.
    */
  def transcribed(params: D2S.Params[Float32]): Seq[(RecordGraph, RecordGraph)] =
    data.samples
      .grouped(TranscribedTogether)
      .flatMap(batch => batch.map(sample => RecordGraph.of(sample.target).copy(edges = Seq.empty)).zip(transcriber(params, batch.map(_.image))))
      .toSeq

  val csv = Metrics.Csv("d2s", setup.corpus, size, Runs.parameters(checkpoints.loadLatest[D2STrainState].get.params), Runs.trainingSeconds(checkpoints.rootPath))
  println(s"writing to ${csv.path}")

  checkpoints.iterations.reverse.foreach: step =>
    println(s"scoring checkpoint $step")
    val drawings = transcribed(checkpoints.load[D2STrainState](step).getOrElse(sys.error(s"no checkpoint $step")).params)
    val measured = Tolerances.map(tolerance => Metrics.Row(step, None, tolerance, drawings.map((target, written) => RecordScoring.score(target, written, tolerance / Canvas))))
    csv.append(measured)
    if step == checkpoints.iterations.last then
      measured.foreach(row => RecordScoring.reportAt(row.tolerance, row.scored))
      println(s"transcripts written to ${Transcripts.write("d2s", setup.corpus, size, step, drawings)}")

/** Plots what a trained model transcribes.
  *
  * A record has no drawing of its own, so it is drawn as the objects it stands for. Note that
  * touching the training split downloads the whole of it on first use.
  */
def plotSetTranscriber(setup: D2SSetup): Unit =
  dimwit.initialize()

  val checkpoints = TensorTreeCheckpointer.latestIn(setup.checkpointRoot).getOrElse(sys.error(s"no training run in ${setup.checkpointRoot}"))
  println(s"reading ${checkpoints.rootPath}")
  val params = checkpoints.loadLatest[D2STrainState].getOrElse(sys.error(s"no checkpoint in ${checkpoints.rootPath}")).params
  val transcriber = Transcriber(Axis[Node] -> setup.nodeSlots)
  val rows = Seq(Split.Validation, Split.Train).flatMap: split =>
    val data = open(setup, split)
    data
      .samples
      .take(3)
      .zipWithIndex
      .map: (sample, index) =>
        val document = greyLevels(sample.image)
        val target = RecordGraph.of(sample.target).copy(edges = Seq.empty)
        val transcribed = transcriber(params, sample.image)
        println(s"${split.fileName} $index target:      ${describe(target)}")
        println(s"${split.fileName} $index transcribed: ${describe(transcribed)}")
        def drawn(record: RecordGraph) = RecordDrawing(record, document, Axis[Channel])
        Seq(
          plots.imagePlot(document, _.title := s"${split.fileName} $index"),
          plots.imagePlot(drawn(target), _.title := s"${split.fileName} $index — record: ${counted(target)}"),
          plots.imagePlot(drawn(transcribed), _.title := s"${split.fileName} $index — transcribed: ${counted(transcribed)}")
        )
      .toSeq

  display(grid(rows))

/** How many drawings are transcribed together. A batch is one traced computation, so this is what
  * the split costs in calls rather than in work.
  */
private val TranscribedTogether = 32

/** Axis of the drawings transcribed together. */
private trait Drawing derives Label

/** Transcribes documents into the set of nodes they hold, one node at a time.
  *
  * Nothing but the document goes in, and every step reads back only what the model itself has
  * taken, so no target record can reach what is written down.
  *
  * A slot's queries each answer with a different remaining node, and the pool is read as a whole:
  * at every slot the answer the model gives the highest log probability is written down — the
  * score of its class plus the score of each coordinate the class places — and never looked back
  * on.
  *
  * A batch of drawings is transcribed in lockstep and as a single traced computation — the
  * encoder, every decoding step and the scorer together — so a batch costs one compiled call
  * rather than one dispatch per operation per step per drawing. `drawings` is how wide that batch
  * is; a call handing over fewer is filled up with a drawing it already holds and read back short
  * again, so that every batch is the same shape. The parameters are an argument, so the one
  * compiled computation serves every batch of every checkpoint.
  *
  * In lockstep means every drawing takes its first node, then its second, and so on for as many
  * slots as a record has. A drawing that has answered [[NodeClass.NoNode]] takes nothing more, and
  * the slots it would have filled hold nothing — which is what a position a record does not reach
  * holds anyway, so the record a drawing ends up with is the one it would have been given had it
  * been transcribed on its own. That is what makes a step
  * the same piece of work whatever the drawings answer, and so something `jit` can compile once
  * and `vmap` can spread over the batch: where a transcription stops becomes a value rather than a
  * branch, and nothing is read back to the host until the whole batch is written down.
  *
  * Each step still re-reads what it has taken so far, since there is no KV cache. That makes every
  * step cost more than it needs to, which is of no consequence here.
  */
class Transcriber(nodes: AxisExtent[Node], drawings: Int = 1):

  private val transcribe = jit: (params: D2S.Params[Float32], documents: Tensor4[Drawing, Width, Height, Channel, Float32]) =>
    written(D2S(params), documents)

  /** The nodes a batch of documents hold, as records that relate none of them. */
  private def written(model: D2S[Float32], documents: Tensor4[Drawing, Width, Height, Channel, Float32]): RecordBatch[Drawing, Node, Relationship] =

    val everyDrawing = documents.shape.extent(Axis[Drawing])

    /** Nothing written yet: every slot of every drawing empty. */
    val nothingWritten =
      val (allNodes, noRelationships) = (Shape2(everyDrawing, nodes), Shape2(everyDrawing, Axis[Relationship] -> 0))
      def nowhere = Tensor(allNodes, VType[Float32]).fill(0f)
      def nothing = Tensor(noRelationships, VType[Int32]).fill(0)
      RecordBatch[Drawing, Node, Relationship](
        nodeClass = Tensor(allNodes, VType[Int32]).fill(NodeClass.NoNode.id),
        startX = nowhere,
        startY = nowhere,
        endX = nowhere,
        endY = nowhere,
        midX = nowhere,
        midY = nowhere,
        edgeClass = nothing,
        subject = nothing,
        obj = nothing
      )

    /** Every drawing still writing down what it is asked for, which at the start is all of them. */
    val allWriting = Tensor1(everyDrawing, VType[Bool]).fill(true)

    val encoded = documents.vmap(Axis[Drawing])(model.encodeDocument)

    /** The likeliest of what the pool answers at one node slot, for every drawing. The slots past
      * what a drawing has written hold nothing, and a prediction reads only the slots before its
      * own, so the answer is the one that slot would have been given on its own.
      */
    def answeredNode(taken: RecordBatch[Drawing, Node, Relationship], slot: Int) =
      zipvmap(Axis[Drawing])(encoded, taken.nodeClass, taken.startX, taken.startY, taken.endX, taken.endY, taken.midX, taken.midY):
        case (document, nodeClass, startX, startY, endX, endY, midX, midY) =>
          val scored = model.logitsPerQuery(document, RecordNodes(nodeClass, startX, startY, endX, endY, midX, midY))
          val (saidClass, saidStartX, saidStartY, saidEndX, saidEndY, saidMidX, saidMidY, score) =
            zipvmap(Axis[PoolQuery])(scored.nodeClass, scored.startX, scored.startY, scored.endX, scored.endY, scored.midX, scored.midY):
              case (nodeClass, startX, startY, endX, endY, midX, midY) =>
                answered(model.nodeHead, NodeLogits(nodeClass, startX, startY, endX, endY, midX, midY))
          val here = Axis[Node].at(slot)
          val likeliest = Axis[PoolQuery].at(score.slice(here).argmax(Axis[PoolQuery]))
          def said[W](answers: Tensor2[PoolQuery, Node, W]) = answers.slice(here).slice(likeliest)
          (said(saidClass), said(saidStartX), said(saidStartY), said(saidEndX), said(saidEndY), said(saidMidX), said(saidMidY))

    /** The slot a step fills, as a mask over the record's slots. */
    def only[L: Label](slots: AxisExtent[L], slot: Int) =
      Tensor1(slots.axis, VType[Bool]).fromArray(Array.tabulate(slots.size)(_ == slot))

    /** One more node slot, filled by every drawing that is still writing nodes. A drawing that
      * answers [[NodeClass.NoNode]] stops there, and the slots it would have filled stay empty.
      */
    def writeNode(taken: RecordBatch[Drawing, Node, Relationship], writing: Tensor1[Drawing, Bool], slot: Int) =
      val (said, startX, startY, endX, endY, midX, midY) = answeredNode(taken, slot)
      val fills = writing and !(said elementEquals_! NodeClass.NoNode.id)
      val here = Shape2(everyDrawing, nodes)
      val filling = only(nodes, slot).broadcastTo(here) and fills.broadcastTo(here)
      def put[W](old: Tensor2[Drawing, Node, W], now: Tensor1[Drawing, W]) =
        where_!(filling, now, old)
      val record = taken.copy(
        nodeClass = put(taken.nodeClass, said),
        startX = put(taken.startX, startX),
        startY = put(taken.startY, startY),
        endX = put(taken.endX, endX),
        endY = put(taken.endY, endY),
        midX = put(taken.midX, midX),
        midY = put(taken.midY, midY)
      )
      (record, fills)

    val (withNodes, _) = (0 until nodes.size).foldLeft((nothingWritten, allWriting)):
      case ((taken, writing), slot) => writeNode(taken, writing, slot)

    withNodes

  def apply(params: D2S.Params[Float32], document: Tensor3[Width, Height, Channel, Float32]): RecordGraph =
    apply(params, Seq(document)).head

  def apply(params: D2S.Params[Float32], documents: Seq[Tensor3[Width, Height, Channel, Float32]]): Seq[RecordGraph] =
    require(documents.nonEmpty, "there is nothing to transcribe")
    require(documents.size <= drawings, s"${documents.size} drawings do not fit in a batch of $drawings")
    val filled = documents.padTo(drawings, documents.last)
    RecordGraph.of(transcribe(params, stack(filled, Axis[Drawing]))).take(documents.size)

private def open(setup: D2SSetup, split: Split) =
  DrawingDataset.open(setup.corpus)(Axis[Width], Axis[Height], Axis[Channel], Axis[Node], Axis[Relationship])(split)

/** What one query answered with at every node slot, and the log probability of that answer. */
def answered(scorer: NodeHead[Float32], logits: NodeLogits[Float32]) =
  val decided = scorer.decide(logits)
  def carries(holds: NodeClass => Boolean) = NodeClass.indicator(VType[Float32])(holds).take(Axis[NodeClasses])(decided.nodeClass)
  val score = chosen(logits.nodeClass) + chosen(logits.startX) + chosen(logits.startY) +
    (chosen(logits.endX) + chosen(logits.endY)) * carries(_.numPoints > 1) +
    (chosen(logits.midX) + chosen(logits.midY)) * carries(_.numPoints > 2)
  (decided.nodeClass, decided.startX, decided.startY, decided.endX, decided.endY, decided.midX, decided.midY, score)

/** The log probability of the value a position's scores are highest for. */
def chosen[Slot: Label, L: Label](logits: Tensor2[Slot, L, Float32]): Tensor1[Slot, Float32] =
  val peak = logits.max(Axis[L])
  peak - (peak + (logits -! peak).exp.sum(Axis[L]).log)

/** How much of a record there is to see, for the header of a drawing of it. */
def counted(record: RecordGraph): String =
  def held(nodeClass: NodeClass, name: String) =
    val count = record.nodes.count(_.nodeClass == nodeClass)
    s"$count $name${if count == 1 then "" else "s"}"
  s"${held(NodeClass.Line, "line")}, ${held(NodeClass.Annotation, "text")}"

def describe(record: RecordGraph): String =
  val nodes = record.nodes.map(node => s"${node.nodeClass}(${node.points.map(point => f"${point.x}%.3f, ${point.y}%.3f").mkString("; ")})")
  val edges = record.edges.map(edge => s"${edge.edgeClass}(${edge.subject}, ${edge.obj})")
  (nodes ++ edges).mkString(", ")
