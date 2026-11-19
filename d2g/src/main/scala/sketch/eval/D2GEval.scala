package d2g.sketch.eval

import d2g.*
import d2g.model.*
import d2g.sketch.config.*
import d2g.sketch.model.*
import d2g.sketch.train.D2GTrainState
import d2s.eval.chosen
import d2s.sketch.eval.answeredAt
import dataset.DrawingDataset
import dataset.DrawingDataset.Split
import dataset.EdgeClass
import dataset.Metrics
import dataset.NodeClass
import dataset.RecordBatch
import dataset.RecordEdges
import dataset.RecordGraph
import dataset.RecordNodes
import dataset.RecordScoring
import dataset.Runs
import dataset.Tolerances
import dataset.Transcripts
import deepwit.attention.KVCache
import deepwit.checkpointing.TensorTreeCheckpointer
import dimwit.*
import dimwit.tensor.Tensor4

import scala.language.implicitConversions

/** Scores the newest run of a setup on a whole split, and writes the metrics as one CSV, and what
  * the last checkpoint wrote down for every drawing as [[Transcripts]]. On the validation split
  * every checkpoint is scored, the last first; the test split is looked at once, by the last alone.
  */
def scoreTranscriber(setup: D2GSetup, size: String, split: Split): Unit =
  dimwit.initialize()

  val checkpoints = TensorTreeCheckpointer.latestIn(setup.checkpointRoot).getOrElse(sys.error(s"no training run in ${setup.checkpointRoot}"))
  println(s"reading ${checkpoints.rootPath}")
  val data = DrawingDataset.open(setup.corpus)(Axis[Width], Axis[Height], Axis[Channel], Axis[Node], Axis[Edge])(split)
  println(s"pool of ${setup.queryPool} queries, ${data.numSamples} drawings\n")

  val transcriber = Transcriber(Axis[Node] -> setup.nodeSlots, Axis[Edge] -> setup.edgeSlots, TranscribedTogether)

  /** Every drawing of the split transcribed by a model, beside the record it was rendered from. */
  def transcribed(params: D2G.Params[Float32]): Seq[(RecordGraph, RecordGraph)] =
    data.samples
      .grouped(TranscribedTogether)
      .flatMap(batch => batch.map(sample => RecordGraph.of(sample.target)).zip(transcriber(params, batch.map(_.image))))
      .toSeq

  val csv = Metrics.Csv("d2g", setup.corpus.name, size, split, Runs.parameters(checkpoints.loadLatest[D2GTrainState].get.params), Runs.trainingSeconds(checkpoints.rootPath))
  println(s"writing to ${csv.path}")

  val scored = split match
    case Split.Test => Seq(checkpoints.iterations.last)
    case _          => checkpoints.iterations.reverse
  scored.foreach: step =>
    println(s"scoring checkpoint $step")
    val drawings = transcribed(checkpoints.load[D2GTrainState](step).getOrElse(sys.error(s"no checkpoint $step")).params)
    val measured = Tolerances.map(tolerance => Metrics.Row(step, None, tolerance, drawings.map((target, written) => RecordScoring.score(target, written, tolerance / setup.corpus.canvas))))
    csv.append(measured)
    if step == checkpoints.iterations.last then
      measured.foreach(row => RecordScoring.reportAt(row.tolerance, row.scored))
      println(s"transcripts written to ${Transcripts.write("d2g", setup.corpus, size, split, step, drawings)}")

/** How many drawings are transcribed together. A batch is one traced computation, so this is what
  * the split costs in calls rather than in work.
  */
private val TranscribedTogether = 32

/** Axis of the drawings transcribed together. */
private trait Drawing derives Label

/** Transcribes drawings: the nodes of each one at a time, as [[d2s.sketch.eval.Transcriber]] writes
  * them, then the relationships between them as [[d2g.floorplan.eval.Transcriber]] writes those of a
  * floor plan, reading the nodes as the node decoder carried them.
  *
  * `drawings` is how many drawings are transcribed in one traced computation; a call handing over
  * fewer is filled up with a drawing it already holds and read back short again.
  */
class Transcriber(nodes: AxisExtent[Node], edges: AxisExtent[Edge], drawings: Int = 1):

  private val jitTranscribe = jit(transcribe)

  private def transcribe(params: D2G.Params[Float32], documents: Tensor4[Drawing, Width, Height, Channel, Float32]): RecordBatch[Drawing, Node, Edge] =
    val model = D2G(params)
    val nodeSlots = Tensor1(nodes.axis).fromRange(0 until nodes.size)
    val edgeSlots = Tensor1(edges.axis).fromRange(0 until edges.size)
    val ((nodeClass, construction, startX, startY, endX, endY, midX, midY), (edgeClass, subject, obj)) = documents.vmap(Axis[Drawing]): document =>
      val encoded = model.encodeDocument(document)
      val documentCache = model.set.read(encoded)
      val (_, (writtenNodes, carried)) = scan(Axis[Node])((model.set.nothingTaken(nodes), Tensor0(true)), nodeSlots):
        case ((taken, writing), slot) =>
          val (node, stillWriting) = answeredAt(model.set, documentCache, taken, slot, writing)
          def one[W](value: Tensor0[W]) = value.prependAxis(Axis[Node])
          val (cls, isConstruction, sx, sy, ex, ey, mx, my) = node
          val (carried, nowTaken) = model.set.take(documentCache, taken, slot, RecordNodes(one(cls), one(isConstruction), one(sx), one(sy), one(ex), one(ey), one(mx), one(my)))
          ((nowTaken, stillWriting), (node, carried))
      val (writtenClass, _, _, _, _, _, _, _) = writtenNodes
      val written = NodeSource(carried, !(writtenClass elementEquals_! NodeClass.NoNode.id))
      val sources = model.readForEdges(encoded, written)
      val (_, writtenEdges) = scan(Axis[Edge])((model.noEdgeTaken(edges), Tensor0(true)), edgeSlots):
        case ((taken, writing), slot) =>
          val (edge, stillWriting) = answeredEdgeAt(model, written, sources, taken, slot, writing)
          def one[W](value: Tensor0[W]) = value.prependAxis(Axis[Edge])
          val (related, sub, ob) = edge
          ((model.takeEdge(written, sources, taken, slot, RecordEdges(one(related), one(sub), one(ob))), stillWriting), edge)
      (writtenNodes, writtenEdges)
    RecordBatch(nodeClass, construction, startX, startY, endX, endY, midX, midY, edgeClass, subject, obj)

  /** The likeliest of what the pool answers at relationship `slot` — the relationship a drawing that
    * is still `writing` takes there, nothing for one that is not — and whether it is still writing
    * after it.
    */
  private def answeredEdgeAt(model: D2G[Float32], nodes: NodeSource[Embedding, Float32], sources: List[EdgeSources[Float32]], taken: List[KVCache[Edge, Float32]], slot: Tensor0[Int32], writing: Tensor0[Bool]) =
    val candidates = model.answerEdgeAt(nodes, sources, taken, slot)
    val said = model.edgeHead.decide(candidates)
    val score = chosen(candidates.edgeClass) + chosen(candidates.subject) + chosen(candidates.obj)
    val likeliest = Axis[Edge].at(score.argmax(Axis[Edge]))
    val stillWriting = writing and !said.edgeClass.slice(likeliest).elementEquals(EdgeClass.NoEdge.id)
    val edge = (
      where(stillWriting, said.edgeClass.slice(likeliest), EdgeClass.NoEdge.id),
      where(stillWriting, said.subject.slice(likeliest), 0),
      where(stillWriting, said.obj.slice(likeliest), 0)
    )
    (edge, stillWriting)

  def apply(params: D2G.Params[Float32], documents: Seq[Tensor3[Width, Height, Channel, Float32]]): Seq[RecordGraph] =
    require(documents.nonEmpty, "there is nothing to transcribe")
    require(documents.size <= drawings, s"${documents.size} drawings do not fit in a batch of $drawings")
    val filled = documents.padTo(drawings, documents.last)
    RecordGraph.of(jitTranscribe(params, stack(filled, Axis[Drawing]))).take(documents.size)
