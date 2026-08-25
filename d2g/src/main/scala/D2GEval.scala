import dataset.Canvas
import dataset.LShapeDataset
import dataset.LShapeDataset.Split
import dataset.NodeClass
import dataset.Outlines
import dataset.Record
import dataset.RecordDrawing
import dataset.RecordGraph
import dataset.RecordScoring
import dataset.Tolerances
import dataset.at
import dataset.report
import deepwit.checkpointing.TensorTreeCheckpointer
import dimwit.*
import dimwit.tensor.Tensor4
import plotwit.*
import viz.PlotTargets.websocket

/** Plots what a trained model transcribes: `sbt "d2g/runMain d2gPlot"`.
  *
  * A record has no drawing of its own, so it is drawn as the objects it stands for, and its
  * relationships are printed. Note that touching the training split downloads 8.6 GB on first use.
  */
@main
def d2gPlot(): Unit =
  dimwit.initialize()

  val checkpoints = TensorTreeCheckpointer.latestIn(D2GCheckpointRoot).getOrElse(sys.error(s"no training run in $D2GCheckpointRoot"))
  println(s"reading ${checkpoints.rootPath}")
  val model = D2G(checkpoints.loadLatest[D2GTrainState].getOrElse(sys.error(s"no checkpoint in ${checkpoints.rootPath}")).params)
  val (nodes, edges) = (Axis[Node] -> NodeSlots, Axis[Edge] -> EdgeSlots)
  val transcriber = Transcriber(model, nodes, edges)
  val rows = Seq(Split.Validation, Split.Train).flatMap: split =>
    val data = open(split)
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

/** Scores a trained model on the whole validation split: `sbt "d2g/runMain d2gEval"`.
  *
  * Every drawing is transcribed autoregressively and the record that comes out is compared with
  * the record it was rendered from, as a record rather than as a sequence. The node lines are what
  * [[detrEval]] reports, on the same records.
  */
@main
def d2gEval(): Unit =
  dimwit.initialize()

  val checkpoints = TensorTreeCheckpointer.latestIn(D2GCheckpointRoot).getOrElse(sys.error(s"no training run in $D2GCheckpointRoot"))
  println(s"reading ${checkpoints.rootPath}")
  val model = D2G(checkpoints.loadLatest[D2GTrainState].getOrElse(sys.error(s"no checkpoint in ${checkpoints.rootPath}")).params)
  val transcriber = Transcriber(model, Axis[Node] -> NodeSlots, Axis[Edge] -> EdgeSlots, TranscribedTogether)
  val data = open(Split.Validation)

  val drawings = data.samples
    .grouped(TranscribedTogether)
    .flatMap(batch => batch.map(sample => RecordGraph.of(sample.target)).zip(transcriber(batch.map(_.image))))
    .toSeq

  val rightLength = drawings.count((target, transcribed) => transcribed.size == target.size)
  report("", "records the right length", rightLength, drawings.length)

  Tolerances.foreach: tolerance =>
    val scored = drawings.map((target, transcribed) => RecordScoring.score(target, transcribed, tolerance / Canvas))
    report(at(tolerance), "nodes transcribed", scored.map(_.nodesFound).sum, scored.map(_.nodes).sum)
    report(at(tolerance), "relationships transcribed", scored.map(_.relationshipsFound).sum, scored.map(_.relationships).sum)
    report(at(tolerance), "records exactly right", scored.count(_.isExact), scored.length)

/** Axis of the drawings transcribed together. */
private trait Drawing derives Label

/** Transcribes documents: the nodes of each one at a time, then the relationships between them.
  *
  * A batch of drawings is transcribed in lockstep and as a single traced computation — the
  * encoder, every decoding step and both scorers together — so a batch costs one compiled call
  * rather than one dispatch per operation per step per drawing. `drawings` is how wide that batch
  * is; a call handing over fewer is filled up with a drawing it already holds and read back short
  * again, so that every batch is the same shape and the computation is compiled once for the
  * whole split.
  *
  * Each step still re-reads what it has taken so far, since there is no KV cache. That makes
  * every step cost more than it needs to, which is of no consequence here: what matters is that
  * the only thing handed to the model is the document, so no target can leak into what is scored.
  */
class Transcriber(model: D2G[Float32], nodes: AxisExtent[Node], edges: AxisExtent[Edge], drawings: Int = 1)
    extends (Tensor3[Width, Height, Channel, Float32] => RecordGraph):

  private val transcribe = jit: (documents: Tensor4[Drawing, Width, Height, Channel, Float32]) =>
    model.predictRecords(documents.vmap(Axis[Drawing])(model.encode), nodes, edges)

  override def apply(document: Tensor3[Width, Height, Channel, Float32]): RecordGraph =
    apply(Seq(document)).head

  def apply(documents: Seq[Tensor3[Width, Height, Channel, Float32]]): Seq[RecordGraph] =
    require(documents.nonEmpty, "there is nothing to transcribe")
    require(documents.size <= drawings, s"${documents.size} drawings do not fit in a batch of $drawings")
    val filled = documents.padTo(drawings, documents.last)
    RecordGraph.of(transcribe(stack(filled, Axis[Drawing]))).take(documents.size)

private def open(split: Split) =
  LShapeDataset.open(Axis[Width], Axis[Height], Axis[Channel], Axis[Node], Axis[Edge])(split)

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
