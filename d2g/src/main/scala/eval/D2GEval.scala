package d2g.eval

import d2g.*
import d2g.model.*
import d2g.train.*
import d2g.config.*
import EdgeHead.EdgeLogits
import NodeHead.NodeLogits
import dataset.Canvas
import dataset.Corpus
import dataset.DrawingDataset
import dataset.DrawingDataset.Split
import dataset.EdgeClass
import dataset.Metrics
import dataset.NodeClass
import dataset.NodeClasses
import dataset.Outlines
import dataset.Record
import dataset.RecordBatch
import dataset.RecordEdges
import dataset.RecordNodes
import dataset.RecordDrawing
import dataset.RecordGraph
import dataset.RecordScoring
import dataset.Runs
import dataset.Tolerances
import dataset.at
import dataset.report
import deepwit.checkpointing.TensorTreeCheckpointer
import dimwit.*
import dimwit.Conversions.given
import dimwit.tensor.Tensor4
import plotwit.*
import viz.PlotTargets.websocket

import scala.language.implicitConversions

/** Scores every checkpoint of the newest run of a setup on the whole validation split, and writes
  * the metrics as one CSV.
  */
def scoreTranscriber(setup: D2GSetup, size: String): Unit =
  dimwit.initialize()

  val checkpoints = TensorTreeCheckpointer.latestIn(setup.checkpointRoot).getOrElse(sys.error(s"no training run in ${setup.checkpointRoot}"))
  println(s"reading ${checkpoints.rootPath}")
  val (nodes, edges) = (Axis[Node] -> setup.nodeSlots, Axis[Edge] -> setup.edgeSlots)
  val data = open(setup, Split.Validation)
  println(s"pool of ${setup.queryPool} queries, ${data.numSamples} drawings\n")

  /** Every drawing of the split transcribed by a model, beside the record it was rendered from. */
  def transcribed(params: D2G.Params[Float32]): Seq[(RecordGraph, RecordGraph)] =
    val transcriber = Transcriber(D2G(params), nodes, edges, TranscribedTogether)
    data.samples
      .grouped(TranscribedTogether)
      .flatMap(batch => batch.map(sample => RecordGraph.of(sample.target)).zip(transcriber(batch.map(_.image))))
      .toSeq

  val rows = checkpoints.iterations.flatMap: step =>
    println(s"scoring checkpoint $step")
    val drawings = transcribed(checkpoints.load[D2GTrainState](step).getOrElse(sys.error(s"no checkpoint $step")).params)
    Tolerances.map(tolerance => Metrics.Row(step, None, tolerance, drawings.map((target, written) => RecordScoring.score(target, written, tolerance / Canvas))))

  rows.filter(_.step == checkpoints.iterations.last).foreach(row => RecordScoring.reportAt(row.tolerance, row.scored))
  val params = checkpoints.loadLatest[D2GTrainState].get.params
  println(s"written to ${Metrics.write("d2g", setup.corpus, size, Runs.parameters(params), Runs.trainingSeconds(checkpoints.rootPath), rows)}")

/** Plots what a trained model transcribes.
  *
  * A record has no drawing of its own, so it is drawn as the objects it stands for, and its
  * relationships are printed. Note that touching the training split downloads the whole of it on
  * first use.
  */
def plotTranscriber(setup: D2GSetup): Unit =
  dimwit.initialize()

  val checkpoints = TensorTreeCheckpointer.latestIn(setup.checkpointRoot).getOrElse(sys.error(s"no training run in ${setup.checkpointRoot}"))
  println(s"reading ${checkpoints.rootPath}")
  val model = D2G(checkpoints.loadLatest[D2GTrainState].getOrElse(sys.error(s"no checkpoint in ${checkpoints.rootPath}")).params)
  val (nodes, edges) = (Axis[Node] -> setup.nodeSlots, Axis[Edge] -> setup.edgeSlots)
  val transcriber = Transcriber(model, nodes, edges)
  val rows = Seq(Split.Validation, Split.Train).flatMap: split =>
    val data = open(setup, split)
    data
      .samples
      .take(3)
      .zipWithIndex
      .map: (sample, index) =>
        val document = Outlines.greyLevels(sample.image)
        val target = RecordGraph.of(sample.target)
        val transcribed = transcriber(sample.image)
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

/** Transcribes documents: the nodes of each one at a time, then the relationships between them.
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
  * encoder, every decoding step and both scorers together — so a batch costs one compiled call
  * rather than one dispatch per operation per step per drawing. `drawings` is how wide that batch
  * is; a call handing over fewer is filled up with a drawing it already holds and read back short
  * again, so that every batch is the same shape and the computation is compiled once for the
  * whole split.
  *
  * In lockstep means every drawing takes its first node, then its second, and so on for as many
  * slots as a record has. A drawing that has answered [[NodeClass.NoNode]] takes nothing more, and
  * the slots it would have filled hold nothing — which is what a position a record does not reach
  * holds anyway, so the record a drawing ends up with is the one it would have been given had it
  * been transcribed on its own. Its relationships follow the same way. That is what makes a step
  * the same piece of work whatever the drawings answer, and so something `jit` can compile once
  * and `vmap` can spread over the batch: where a transcription stops becomes a value rather than a
  * branch, and nothing is read back to the host until the whole batch is written down.
  *
  * Each step still re-reads what it has taken so far, since there is no KV cache. That makes every
  * step cost more than it needs to, which is of no consequence here.
  */
class Transcriber(model: D2G[Float32], nodes: AxisExtent[Node], edges: AxisExtent[Edge], drawings: Int = 1)
    extends (Tensor3[Width, Height, Channel, Float32] => RecordGraph):

  private val transcribe = jit: (documents: Tensor4[Drawing, Width, Height, Channel, Float32]) =>
    written(documents)

  /** The records a batch of documents hold. */
  private def written(documents: Tensor4[Drawing, Width, Height, Channel, Float32]): RecordBatch[Drawing, Node, Edge] =

    val everyDrawing = documents.shape.extent(Axis[Drawing])

    /** Nothing written yet: every slot of every drawing empty. */
    val nothingWritten =
      val (allNodes, allEdges) = (Shape2(everyDrawing, nodes), Shape2(everyDrawing, edges))
      def nowhere = Tensor(allNodes, VType[Float32]).fill(0f)
      def nothing = Tensor(allEdges, VType[Int32]).fill(0)
      RecordBatch[Drawing, Node, Edge](
        nodeClass = Tensor(allNodes, VType[Int32]).fill(NodeClass.NoNode.id),
        startX = nowhere,
        startY = nowhere,
        endX = nowhere,
        endY = nowhere,
        edgeClass = Tensor(allEdges, VType[Int32]).fill(EdgeClass.NoEdge.id),
        subject = nothing,
        obj = nothing
      )

    /** Every drawing still writing down what it is asked for, which at the start is all of them.
      * The nodes and the relationships are two stages, and a drawing writes both.
      */
    val allWriting = Tensor1(everyDrawing, VType[Bool]).fill(true)

    val encoded = documents.vmap(Axis[Drawing])(model.encodeDocument)

    /** The likeliest of what the pool answers at one node slot, for every drawing. The slots past
      * what a drawing has written hold nothing, and a prediction reads only the slots before its
      * own, so the answer is the one that slot would have been given on its own.
      */
    def answeredNode(taken: RecordBatch[Drawing, Node, Edge], slot: Int) =
      zipvmap(Axis[Drawing])(encoded, taken.nodeClass, taken.startX, taken.startY, taken.endX, taken.endY, taken.edgeClass, taken.subject, taken.obj):
        case (document, nodeClass, startX, startY, endX, endY, edgeClass, subject, obj) =>
          val scored = model.logitsPerQuery(document, Record(RecordNodes(nodeClass, startX, startY, endX, endY), RecordEdges(edgeClass, subject, obj))).nodes
          val (saidClass, saidStartX, saidStartY, saidEndX, saidEndY, score) =
            zipvmap(Axis[PoolQuery])(scored.nodeClass, scored.startX, scored.startY, scored.endX, scored.endY):
              case (nodeClass, startX, startY, endX, endY) =>
                answered(model.nodeHead, NodeLogits(nodeClass, startX, startY, endX, endY))
          val here = Axis[Node].at(slot)
          val likeliest = Axis[PoolQuery].at(score.slice(here).argmax)
          (saidClass.slice(here).slice(likeliest), saidStartX.slice(here).slice(likeliest), saidStartY.slice(here).slice(likeliest), saidEndX.slice(here).slice(likeliest), saidEndY.slice(here).slice(likeliest))

    /** The same for a relationship slot. */
    def answeredEdge(taken: RecordBatch[Drawing, Node, Edge], slot: Int) =
      zipvmap(Axis[Drawing])(encoded, taken.nodeClass, taken.startX, taken.startY, taken.endX, taken.endY, taken.edgeClass, taken.subject, taken.obj):
        case (document, nodeClass, startX, startY, endX, endY, edgeClass, subject, obj) =>
          val scored = model.logitsPerQuery(document, Record(RecordNodes(nodeClass, startX, startY, endX, endY), RecordEdges(edgeClass, subject, obj))).edges
          val (saidClass, saidSubject, saidObj, score) =
            zipvmap(Axis[PoolQuery])(scored.edgeClass, scored.subject, scored.obj):
              case (edgeClass, subject, obj) => answered(model.edgeHead, EdgeLogits(edgeClass, subject, obj))
          val here = Axis[Edge].at(slot)
          val likeliest = Axis[PoolQuery].at(score.slice(here).argmax)
          (saidClass.slice(here).slice(likeliest), saidSubject.slice(here).slice(likeliest), saidObj.slice(here).slice(likeliest))

    /** The slot a step fills, as a mask over the record's slots. */
    def only[L: Label](slots: AxisExtent[L], slot: Int) =
      Tensor1(slots.axis, VType[Bool]).fromArray(Array.tabulate(slots.size)(_ == slot))

    /** One more node slot, filled by every drawing that is still writing nodes. A drawing that
      * answers [[NodeClass.NoNode]] stops there, and the slots it would have filled stay empty.
      */
    def writeNode(taken: RecordBatch[Drawing, Node, Edge], writing: Tensor1[Drawing, Bool], slot: Int) =
      val (said, startX, startY, endX, endY) = answeredNode(taken, slot)
      val fills = writing and !(said elementEquals_! NodeClass.NoNode.id)
      val here = Shape2(everyDrawing, nodes)
      val filling = only(nodes, slot).broadcastTo(here) and fills.broadcastTo(here)
      def put[W](old: Tensor2[Drawing, Node, W], now: Tensor1[Drawing, W]) =
        where(filling, now.broadcastTo(here), old)
      val record = taken.copy(
        nodeClass = put(taken.nodeClass, said),
        startX = put(taken.startX, startX),
        startY = put(taken.startY, startY),
        endX = put(taken.endX, endX),
        endY = put(taken.endY, endY)
      )
      (record, fills)

    /** One more relationship slot, filled the same way the nodes were. */
    def writeEdge(taken: RecordBatch[Drawing, Node, Edge], writing: Tensor1[Drawing, Bool], slot: Int) =
      val (said, subject, obj) = answeredEdge(taken, slot)
      val fills = writing and !(said elementEquals_! EdgeClass.NoEdge.id)
      val here = Shape2(everyDrawing, edges)
      val filling = only(edges, slot).broadcastTo(here) and fills.broadcastTo(here)
      def put[W](old: Tensor2[Drawing, Edge, W], now: Tensor1[Drawing, W]) =
        where(filling, now.broadcastTo(here), old)
      val record = taken.copy(
        edgeClass = put(taken.edgeClass, said),
        subject = put(taken.subject, subject),
        obj = put(taken.obj, obj)
      )
      (record, fills)

    val (withNodes, _) = (0 until nodes.size).foldLeft((nothingWritten, allWriting)):
      case ((taken, writing), slot) => writeNode(taken, writing, slot)

    val (withEdges, _) = (0 until edges.size).foldLeft((withNodes, allWriting)):
      case ((taken, writing), slot) => writeEdge(taken, writing, slot)

    withEdges

  override def apply(document: Tensor3[Width, Height, Channel, Float32]): RecordGraph =
    apply(Seq(document)).head

  def apply(documents: Seq[Tensor3[Width, Height, Channel, Float32]]): Seq[RecordGraph] =
    require(documents.nonEmpty, "there is nothing to transcribe")
    require(documents.size <= drawings, s"${documents.size} drawings do not fit in a batch of $drawings")
    val filled = documents.padTo(drawings, documents.last)
    RecordGraph.of(transcribe(stack(filled, Axis[Drawing]))).take(documents.size)

private def open(setup: D2GSetup, split: Split) =
  DrawingDataset.open(setup.corpus)(Axis[Width], Axis[Height], Axis[Channel], Axis[Node], Axis[Edge])(split)

/** What one query answered with at every node slot, and the log probability of that answer. */
private def answered(scorer: NodeHead[Float32], logits: NodeLogits[Float32]) =
  val decided = scorer.decide(logits)
  val carriesEnd = NodeClass.indicator(VType[Float32])(_.numPoints > 1).take(Axis[NodeClasses])(decided.nodeClass)
  val score = chosen(logits.nodeClass) + chosen(logits.startX) + chosen(logits.startY) +
    (chosen(logits.endX) + chosen(logits.endY)) * carriesEnd
  (decided.nodeClass, decided.startX, decided.startY, decided.endX, decided.endY, score)

/** The same for a relationship. */
private def answered(scorer: EdgeHead[Float32], logits: EdgeLogits[Float32]) =
  val decided = scorer.decide(logits)
  (decided.edgeClass, decided.subject, decided.obj, chosen(logits.edgeClass) + chosen(logits.subject) + chosen(logits.obj))

/** The log probability of the value a position's scores are highest for. */
private def chosen[Slot: Label, L: Label](logits: Tensor2[Slot, L, Float32]): Tensor1[Slot, Float32] =
  val peak = logits.max(Axis[L])
  peak - (peak + (logits - peak.broadcastTo(logits.shape)).exp.sum(Axis[L]).log)

/** How much of a record there is to see, for the header of a drawing of it. */
private def counted(record: RecordGraph): String =
  def held(nodeClass: NodeClass, name: String) =
    val count = record.nodes.count(_.nodeClass == nodeClass)
    s"$count $name${if count == 1 then "" else "s"}"
  s"${held(NodeClass.Line, "line")}, ${held(NodeClass.Annotation, "text")}"

private def describe(record: RecordGraph): String =
  val nodes = record.nodes.map(node => s"${node.nodeClass}(${node.points.map(point => f"${point.x}%.3f, ${point.y}%.3f").mkString("; ")})")
  val edges = record.edges.map(edge => s"${edge.edgeClass}(${edge.subject}, ${edge.obj})")
  (nodes ++ edges).mkString(", ")
