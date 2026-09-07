import EdgeScorer.EdgeLogits
import NodeScorer.NodeLogits
import dataset.Canvas
import dataset.DrawingDataset
import dataset.DrawingDataset.Split
import dataset.EdgeClass
import dataset.NodeClass
import dataset.NodeClasses
import dataset.Record
import dataset.RecordBatch
import dataset.RecordEdge
import dataset.RecordGraph
import dataset.RecordNode
import dataset.RecordScoring
import dataset.Tolerances
import dataset.report
import deepwit.checkpointing.TensorTreeCheckpointer
import dimwit.*
import dimwit.tensor.Tensor4

/** How many partial records are carried along per drawing.
  *
  * A slot's queries each answer with a different remaining node, so at every step a transcription
  * has as many ways to go on as the pool is wide. [[d2gEval]] follows one query all the way down
  * and commits to its fork every time; here the forks are kept, each step branches them, and the
  * least likely are dropped.
  *
  * A record's likelihood is the sum of what it cost to write down — for every node, the score its
  * class was given plus the score of each coordinate the class places, all as log probabilities. So
  * a step that had to guess is paid for, and a search that took a cheaper route early can be
  * overtaken by one that stayed sure of itself. One is greedy: the best answer the pool offers is
  * taken at every step and never looked back on.
  */
private val BeamWidth = 4

/** How many drawings are searched at once. Every drawing carries [[BeamWidth]] partial records, so
  * the model reads this many times that on every step.
  */
private val SearchedTogether = 16

/** Axis of the partial records carried along, drawings and their beams flattened together. */
private trait Carried derives Label

/** One partial record and what it cost to write down. */
private case class Beam(record: RecordGraph, score: Float, nodesEnded: Boolean, edgesEnded: Boolean):
  def ended: Boolean = nodesEnded && edgesEnded

def d2gBeamSearch(setup: D2GSetup, width: Int = BeamWidth): Unit =
  dimwit.initialize()

  val checkpoints = TensorTreeCheckpointer.latestIn(setup.checkpointRoot).getOrElse(sys.error(s"no training run in ${setup.checkpointRoot}"))
  println(s"reading ${checkpoints.rootPath}")
  val params = checkpoints.loadLatest[D2GTrainState].getOrElse(sys.error(s"no checkpoint in ${checkpoints.rootPath}")).params
  val model = D2G(params)
  val (nodes, edges) = (Axis[Node] -> setup.nodeSlots, Axis[Edge] -> setup.edgeSlots)
  val carried = Axis[Carried] -> (SearchedTogether * width)

  /** What every query of the pool answers at a slot, and what each answer is worth — one reading
    * for the whole pool, since the queries never read each other.
    */
  val ask = jit: (
      images: Tensor4[Carried, Width, Height, Channel, Float32],
      taken: RecordBatch[Carried, Node, Edge]
  ) =>
    zipvmap(Axis[Carried])(images, taken.nodeClass, taken.startX, taken.startY, taken.endX, taken.endY, taken.edgeClass, taken.subject, taken.obj):
      case (image, nc, sx, sy, ec2, ey, ec, su, ob) =>
        val scored = model.logitsPerQuery(image, Record(dataset.RecordNodes(nc, sx, sy, ec2, ey), dataset.RecordEdges(ec, su, ob)))
        val (nodeClass, startX, startY, endX, endY, nodeScore) =
          zipvmap(Axis[Query])(scored.nodes.nodeClass, scored.nodes.startX, scored.nodes.startY, scored.nodes.endX, scored.nodes.endY):
            case (nodeClass, startX, startY, endX, endY) =>
              answeredNode(model.nodeScorer, NodeLogits(nodeClass, startX, startY, endX, endY))
        val (edgeClass, subject, obj, edgeScore) =
          zipvmap(Axis[Query])(scored.edges.edgeClass, scored.edges.subject, scored.edges.obj):
            case (edgeClass, subject, obj) => answeredEdge(model.edgeScorer, EdgeLogits(edgeClass, subject, obj))
        (nodeClass, startX, startY, endX, endY, nodeScore, edgeClass, subject, obj, edgeScore)

  val data = DrawingDataset.open(setup.corpus)(Axis[Width], Axis[Height], Axis[Channel], Axis[Node], Axis[Edge])(Split.Validation)
  println(s"beam of $width over ${setup.queryPool} queries, ${data.numSamples} drawings\n")

  val scored = data.samples
    .grouped(SearchedTogether)
    .filter(_.size == SearchedTogether)
    .map: batch =>
      searched(batch.map(_.image), nodes, edges, carried, width, setup.queryPool, ask).zip(batch.map(sample => RecordGraph.of(sample.target)))
    .flatMap(found => found.map((written, target) => (target, written)))
    .toSeq

  Tolerances.foreach: tolerance =>
    RecordScoring.reportAt(tolerance, scored.map((target, written) => RecordScoring.score(target, written, tolerance / Canvas)))

/** The best record the search finds for every drawing of a batch. */
private def searched(
    documents: Seq[Tensor3[Width, Height, Channel, Float32]],
    nodes: AxisExtent[Node],
    edges: AxisExtent[Edge],
    carried: AxisExtent[Carried],
    width: Int,
    pool: Int,
    ask: (Tensor4[Carried, Width, Height, Channel, Float32], RecordBatch[Carried, Node, Edge]) => (
        Tensor3[Carried, Query, Node, Int32],
        Tensor3[Carried, Query, Node, Float32],
        Tensor3[Carried, Query, Node, Float32],
        Tensor3[Carried, Query, Node, Float32],
        Tensor3[Carried, Query, Node, Float32],
        Tensor3[Carried, Query, Node, Float32],
        Tensor3[Carried, Query, Edge, Int32],
        Tensor3[Carried, Query, Edge, Int32],
        Tensor3[Carried, Query, Edge, Int32],
        Tensor3[Carried, Query, Edge, Float32]
    )
): Seq[RecordGraph] =
  var beams = documents.map(_ => Seq(Beam(RecordGraph(Seq.empty, Seq.empty), 0f, false, false)))
  val images = stack(documents.flatMap(document => Seq.fill(width)(document)), carried.axis)

  /** What the queries answer for every partial record, read back to the host once per step — a
    * slice at a time would be thousands of transfers for one step.
    */
  def asked(): (NodeAnswers, EdgeAnswers) =
    val laid = beams.flatMap(drawing => drawing.padTo(width, drawing.last).map(_.record))
    val said = ask(images, RecordBatch.of(laid, carried.axis, nodes, edges))
    (
      NodeAnswers(said._1.toArray, said._2.toArray, said._3.toArray, said._4.toArray, said._5.toArray, said._6.toArray),
      EdgeAnswers(said._7.toArray, said._8.toArray, said._9.toArray, said._10.toArray)
    )

  // -- the nodes, one slot at a time -------------------------------------------------------------
  (0 until nodes.size).foreach: _ =>
    if beams.exists(_.exists(!_.nodesEnded)) then
      val said = asked()._1
      beams = beams.zipWithIndex.map: (drawing, index) =>
        val grown = drawing.zipWithIndex.flatMap: (beam, slot)  =>
          if beam.nodesEnded then Seq(beam)
          else
            val at = index * width + slot
            (0 until pool).map: query =>
              val position = beam.record.nodes.length
              val nodeClass = NodeClass.fromId(said.nodeClass(at)(query)(position))
              val worth = said.score(at)(query)(position)
              if !nodeClass.isNode then beam.copy(score = beam.score + worth, nodesEnded = true)
              else
                val points = Seq(
                  dataset.Point(said.startX(at)(query)(position), said.startY(at)(query)(position)),
                  dataset.Point(said.endX(at)(query)(position), said.endY(at)(query)(position))
                ).take(nodeClass.numPoints)
                beam.copy(
                  record = beam.record.copy(nodes = beam.record.nodes :+ RecordNode(nodeClass, points)),
                  score = beam.score + worth,
                  nodesEnded = position + 1 >= nodes.size - 1
                )
        grown.sortBy(-_.score).take(width)

  // -- then the relationships between them -------------------------------------------------------
  (0 until edges.size).foreach: _ =>
    if beams.exists(_.exists(!_.edgesEnded)) then
      val said = asked()._2
      beams = beams.zipWithIndex.map: (drawing, index) =>
        val grown = drawing.zipWithIndex.flatMap: (beam, slot) =>
          if beam.edgesEnded then Seq(beam)
          else
            val at = index * width + slot
            (0 until pool).map: query =>
              val position = beam.record.edges.length
              val edgeClass = EdgeClass.fromId(said.edgeClass(at)(query)(position))
              val worth = said.score(at)(query)(position)
              if !edgeClass.isEdge then beam.copy(score = beam.score + worth, edgesEnded = true)
              else
                beam.copy(
                  record = beam.record.copy(edges = beam.record.edges :+ RecordEdge(edgeClass, said.subject(at)(query)(position), said.obj(at)(query)(position))),
                  score = beam.score + worth,
                  edgesEnded = position + 1 >= edges.size - 1
                )
        grown.sortBy(-_.score).take(width)

  beams.map(_.maxBy(_.score).record)

/** Everything the queries answered for a batch of partial records, on the host. */
private case class NodeAnswers(
    nodeClass: Array[Array[Array[Int]]],
    startX: Array[Array[Array[Float]]],
    startY: Array[Array[Array[Float]]],
    endX: Array[Array[Array[Float]]],
    endY: Array[Array[Array[Float]]],
    score: Array[Array[Array[Float]]]
)

private case class EdgeAnswers(
    edgeClass: Array[Array[Array[Int]]],
    subject: Array[Array[Array[Int]]],
    obj: Array[Array[Array[Int]]],
    score: Array[Array[Array[Float]]]
)

/** What one query answered with at every node slot, and the log probability of that answer. */
private def answeredNode(scorer: NodeScorer[Float32], logits: NodeLogits[Float32]) =
  val decided = scorer.decide(logits)
  val carriesEnd = NodeClass.indicator(VType[Float32])(_.numPoints > 1).take(Axis[NodeClasses])(decided.nodeClass)
  val score = chosen(logits.nodeClass) + chosen(logits.startX) + chosen(logits.startY) +
    (chosen(logits.endX) + chosen(logits.endY)) * carriesEnd
  (decided.nodeClass, decided.startX, decided.startY, decided.endX, decided.endY, score)

/** The same for a relationship. */
private def answeredEdge(scorer: EdgeScorer[Float32], logits: EdgeLogits[Float32]) =
  val decided = scorer.decide(logits)
  (decided.edgeClass, decided.subject, decided.obj, chosen(logits.edgeClass) + chosen(logits.subject) + chosen(logits.obj))

/** The log probability of the value a position's scores are highest for. */
private def chosen[Slot: Label, L: Label](logits: Tensor2[Slot, L, Float32]): Tensor1[Slot, Float32] =
  val peak = logits.max(Axis[L])
  peak - (peak + (logits - peak.broadcastTo(logits.shape)).exp.sum(Axis[L]).log)

@main
def d2gLShapeBeamSearch(): Unit = d2gBeamSearch(D2GLShape)

@main
def d2gRectilinear6to18BeamSearch(): Unit = d2gBeamSearch(D2GRectilinear6to18)

/** The same search with nothing kept open: every step takes the best the queries offer. */
@main
def d2gRectilinear6to18BeamSearchGreedy(): Unit = d2gBeamSearch(D2GRectilinear6to18, width = 1)

@main
def d2gLShapeBeamSearchGreedy(): Unit = d2gBeamSearch(D2GLShape, width = 1)
