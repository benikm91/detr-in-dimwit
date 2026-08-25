import dataset.Canvas
import dataset.LShapeDataset
import dataset.LShapeDataset.Split
import dataset.NodeClass
import dataset.Objects
import dataset.Outlines
import dataset.Record
import dataset.RecordGraph
import dataset.RecordScoring
import dataset.Tolerances
import dataset.at
import dataset.report
import deepwit.checkpointing.TensorTreeCheckpointer
import dimwit.*
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
        def drawn(record: RecordGraph) = Objects.of(record.record(nodes, edges)).detection
        Seq(
          plots.imagePlot(document, _.title := s"${split.fileName} $index"),
          plots.imagePlot(Outlines(document, drawn(target)), _.title := s"${split.fileName} $index — record"),
          plots.imagePlot(Outlines(document, drawn(transcribed)), _.title := s"${split.fileName} $index — transcribed")
        )
      .toSeq

  display(grid(rows))

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
  val transcriber = Transcriber(model, Axis[Node] -> NodeSlots, Axis[Edge] -> EdgeSlots)
  val data = open(Split.Validation)

  val drawings = data.samples.map(sample => (RecordGraph.of(sample.target), transcriber(sample.image))).toSeq

  val rightLength = drawings.count((target, transcribed) => transcribed.size == target.size)
  report("", "records the right length", rightLength, drawings.length)

  Tolerances.foreach: tolerance =>
    val scored = drawings.map((target, transcribed) => RecordScoring.score(target, transcribed, tolerance / Canvas))
    report(at(tolerance), "nodes transcribed", scored.map(_.nodesFound).sum, scored.map(_.nodes).sum)
    report(at(tolerance), "relationships transcribed", scored.map(_.relationshipsFound).sum, scored.map(_.relationships).sum)
    report(at(tolerance), "records exactly right", scored.count(_.isExact), scored.length)

/** Transcribes a document: its nodes one at a time, then the relationships between them.
  *
  * The document is encoded once and read at every step; each stage re-reads what it has taken so
  * far every time, since there is no KV cache. That makes every step cost more than it needs to,
  * which is of no consequence here: what matters is that the only thing handed to the model is
  * the document, so no target can leak into what is scored.
  */
class Transcriber(model: D2G[Float32], nodes: AxisExtent[Node], edges: AxisExtent[Edge])
    extends (Tensor3[Width, Height, Channel, Float32] => RecordGraph):

  private val encode = jit(model.encode)

  override def apply(document: Tensor3[Width, Height, Channel, Float32]): RecordGraph =
    RecordGraph.of(model.predictRecord(encode(document), nodes, edges))

private def open(split: Split) =
  LShapeDataset.open(Axis[Width], Axis[Height], Axis[Channel], Axis[Node], Axis[Edge])(split)

private def describe(record: RecordGraph): String =
  val nodes = record.nodes.map(node => s"${node.nodeClass}(${node.points.map(point => f"${point.x}%.3f, ${point.y}%.3f").mkString("; ")})")
  val edges = record.edges.map(edge => s"${edge.edgeClass}(${edge.subject}, ${edge.obj})")
  (nodes ++ edges).mkString(", ")
