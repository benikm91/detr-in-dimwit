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
import dimwit.*
import plotwit.*
import viz.PlotTargets.websocket

/** Plots what a trained model transcribes: `sbt "d2g/runMain d2gPlot"`.
  *
  * A record has no drawing of its own, so it is drawn as the objects it stands for, and its
  * relationships are printed. Note that touching the training split downloads 8.6 GB on first use.
  */
@main
def d2gPlot(run: String*): Unit =
  dimwit.initialize()

  val nodes = Axis[Node] -> RecordNodes
  val transcriber = Transcriber(load(run), nodes)
  val rows = Seq(Split.Validation, Split.Train).flatMap: split =>
    val data = open(split)
    data
      .samples()
      .take(3)
      .zipWithIndex
      .map: (sample, index) =>
        val document = Outlines.greyLevels(sample.image)
        val target = RecordGraph.of(sample.target)
        val transcribed = transcriber(sample.image)
        println(s"${split.fileName} $index target:      ${describe(target)}")
        println(s"${split.fileName} $index transcribed: ${describe(transcribed)}")
        def drawn(record: RecordGraph) = Objects.of(record.record(nodes), data.geometry).detection
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
def d2gEval(run: String*): Unit =
  dimwit.initialize()

  val nodes = Axis[Node] -> RecordNodes
  val transcriber = Transcriber(load(run), nodes)
  val data = open(Split.Validation)

  val drawings = data.samples().map(sample => (RecordGraph.of(sample.target), transcriber(sample.image))).toSeq

  val rightLength = drawings.count((target, transcribed) => transcribed.size == target.size)
  report("", "records the right length", rightLength, drawings.length)

  Tolerances.foreach: tolerance =>
    val scored = drawings.map((target, transcribed) => RecordScoring.score(target, transcribed, tolerance / data.imageWidth))
    report(at(tolerance), "nodes transcribed", scored.map(_.nodesFound).sum, scored.map(_.nodes).sum)
    report(at(tolerance), "relationships transcribed", scored.map(_.relationshipsFound).sum, scored.map(_.relationships).sum)
    report(at(tolerance), "records exactly right", scored.count(_.isExact), scored.length)

/** Transcribes a document, one remaining node at a time.
  *
  * The document is encoded once and read at every step. There is no KV cache, so the decoder
  * re-reads the whole sequence every time — slow, and no different. The loop runs to the last slot
  * either way and the answer is cut where the model said the record ends.
  */
class Transcriber(model: D2G[Float32], nodes: AxisExtent[Node])
    extends (Tensor3[Width, Height, Channel, Float32] => RecordGraph):

  private val encode = jit(model.encode)
  private val advance = jit: (document: Tensor2[D2G.Patch, Embedding, Float32], taken: Record[Node], position: Tensor0[Int32]) =>
    taken.taking(model.step(document, taken), position)

  override def apply(document: Tensor3[Width, Height, Channel, Float32]): RecordGraph =
    val encoded = encode(document)
    val empty = RecordGraph(Seq.empty, Seq.empty).record(nodes)
    RecordGraph.of(upToStop((0 until nodes.size - 1).foldLeft(empty)((taken, at) => advance(encoded, taken, Tensor0(at)))))

  private def upToStop(record: Record[Node]): Record[Node] =
    val stop = record.nodeClass.toArray.indexWhere(_ == NodeClass.NoNode.id) match
      case -1 => nodes.size
      case at => at
    val kept = Axis[Node].at(0 until stop)
    Record(record.nodeClass.slice(kept), record.xs.slice(kept), record.ys.slice(kept), record.links.slice(kept))

extension (record: Record[Node])
  /** The record with the node `decided` settled on at `position` taken into it. */
  def taking(decided: Record[Node], position: Tensor0[Int32]): Record[Node] =
    val at = Axis[Node].at(position)
    Record(
      nodeClass = record.nodeClass.set(at)(decided.nodeClass.slice(at)),
      xs = record.xs.set(at)(decided.xs.slice(at)),
      ys = record.ys.set(at)(decided.ys.slice(at)),
      links = record.links.set(at)(decided.links.slice(at))
    )

private def open(split: Split) =
  LShapeDataset.open(Axis[Width], Axis[Height], Axis[Channel], Axis[Node])(
    split,
    LShapeDataset.Config(maxNumNodes = Some(RecordNodes))
  )

private def describe(record: RecordGraph): String =
  val nodes = record.nodes.map(node => s"${node.nodeClass}(${node.points.map(point => f"${point.x}%.3f, ${point.y}%.3f").mkString("; ")})")
  val edges = record.edges.map(edge => s"${edge.edgeClass}(${edge.subject}, ${edge.obj})")
  (nodes ++ edges).mkString(", ")

private def load(): D2G[Float32] =
  val checkpointer = TensorTreeCheckpointer.latestIn("out/d2g").getOrElse(sys.error(s"Nothing to load: $runRoot holds no run yet. Train first."))
  println(s"reading ${checkpointer.rootPath}")
  D2G(checkpointer.loadLatest[D2GTrainState].get.params)
