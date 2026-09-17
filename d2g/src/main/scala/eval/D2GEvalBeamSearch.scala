package d2g.eval

import d2g.*
import d2g.model.*
import d2g.train.*
import d2g.config.*
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
import dataset.Runs
import dataset.Metrics
import dataset.RecordNode
import dataset.RecordScoring
import dataset.Tolerances
import dataset.report
import deepwit.checkpointing.TensorTreeCheckpointer
import dimwit.*
import dimwit.tensor.Tensor4

/** How many drawings are searched at once. Every drawing carries as many partial records as the
  * search is wide, so the model reads this many times that on every step.
  */
private val SearchedTogether = 16

/** Axis of the partial records carried along, drawings and their beams flattened together. */
private trait Carried derives Label

/** One partial record and what it cost to write down. */
private case class Beam(record: RecordGraph, score: Float, nodesEnded: Boolean, edgesEnded: Boolean):
  def ended: Boolean = nodesEnded && edgesEnded

/** Scores the newest run of a setup on the whole validation split, transcribing by a search over
  * what the queries answer.
  *
  * A slot's queries each answer with a different remaining node, so at every step a transcription
  * has as many ways to go on as the pool is wide. `width` is how many partial records are carried
  * along per drawing: each step branches them, and the least likely are dropped. A record's
  * likelihood is the sum of what it cost to write down — for every node, the score its class was
  * given plus the score of each coordinate the class places, all as log probabilities.
  *
  * A width of one is greedy: the best answer the pool offers is taken at every step and never
  * looked back on.
  */
def scoreTranscriber(setup: D2GSetup, size: String, width: Int): Unit =
  dimwit.initialize()

  val checkpoints = TensorTreeCheckpointer.latestIn(setup.checkpointRoot).getOrElse(sys.error(s"no training run in ${setup.checkpointRoot}"))
  println(s"reading ${checkpoints.rootPath}")
  val (nodes, edges) = (Axis[Node] -> setup.nodeSlots, Axis[Edge] -> setup.edgeSlots)
  val carried = Axis[Carried] -> (SearchedTogether * width)
  val data = DrawingDataset.open(setup.corpus)(Axis[Width], Axis[Height], Axis[Channel], Axis[Node], Axis[Edge])(Split.Validation)
  println(s"beam of $width over ${setup.queryPool} queries, ${data.numSamples} drawings\n")

  /** Every drawing of the split transcribed by a model, beside the record it was rendered from. */
  def transcribed(params: D2G.Params[Float32]): Seq[(RecordGraph, RecordGraph)] =
    val model = D2G(params)

    /** What every query of the pool answers at a slot, and what each answer is worth — one reading
      * for the whole pool, since the queries never read each other.
      */
    val encode = jit: (images: Tensor4[Carried, Width, Height, Channel, Float32]) =>
      images.vmap(Axis[Carried])(model.encodeDocument)

    val ask = jit: (
        encoded: Tensor3[Carried, Patch, Embedding, Float32],
        taken: RecordBatch[Carried, Node, Edge]
    ) =>
      zipvmap(Axis[Carried])(encoded, taken.nodeClass, taken.startX, taken.startY, taken.endX, taken.endY, taken.edgeClass, taken.subject, taken.obj):
        case (document, nc, sx, sy, ec2, ey, ec, su, ob) =>
          val scored = model.logitsPerQuery(document, Record(dataset.RecordNodes(nc, sx, sy, ec2, ey), dataset.RecordEdges(ec, su, ob)))
          val (nodeClass, startX, startY, endX, endY, nodeScore) =
            zipvmap(Axis[Query])(scored.nodes.nodeClass, scored.nodes.startX, scored.nodes.startY, scored.nodes.endX, scored.nodes.endY):
              case (nodeClass, startX, startY, endX, endY) =>
                answeredNode(model.nodeScorer, NodeLogits(nodeClass, startX, startY, endX, endY))
          val (edgeClass, subject, obj, edgeScore) =
            zipvmap(Axis[Query])(scored.edges.edgeClass, scored.edges.subject, scored.edges.obj):
              case (edgeClass, subject, obj) => answeredEdge(model.edgeScorer, EdgeLogits(edgeClass, subject, obj))
          (nodeClass, startX, startY, endX, endY, nodeScore, edgeClass, subject, obj, edgeScore)

    data.samples
      .grouped(SearchedTogether)
      .filter(_.size == SearchedTogether)
      .map: batch =>
        searched(batch.map(_.image), nodes, edges, carried, width, setup.queryPool, encode, ask).zip(batch.map(sample => RecordGraph.of(sample.target)))
      .flatMap(found => found.map((written, target) => (target, written)))
      .toSeq

  val rows = checkpoints.iterations.flatMap: step =>
    println(s"scoring checkpoint $step")
    val drawings = transcribed(checkpoints.load[D2GTrainState](step).getOrElse(sys.error(s"no checkpoint $step")).params)
    Tolerances.map(tolerance => Metrics.Row(step, None, tolerance, drawings.map((target, written) => RecordScoring.score(target, written, tolerance / Canvas))))

  rows.filter(_.step == checkpoints.iterations.last).foreach(row => RecordScoring.reportAt(row.tolerance, row.scored))
  val params = checkpoints.loadLatest[D2GTrainState].get.params
  println(s"written to ${Metrics.write("d2g", setup.corpus, size, Runs.parameters(params), Runs.trainingSeconds(checkpoints.rootPath), rows)}")

/** The best record the search finds for every drawing of a batch. */
private def searched(
    documents: Seq[Tensor3[Width, Height, Channel, Float32]],
    nodes: AxisExtent[Node],
    edges: AxisExtent[Edge],
    carried: AxisExtent[Carried],
    width: Int,
    pool: Int,
    encode: Tensor4[Carried, Width, Height, Channel, Float32] => Tensor3[Carried, Patch, Embedding, Float32],
    ask: (Tensor3[Carried, Patch, Embedding, Float32], RecordBatch[Carried, Node, Edge]) => (
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
  // Every beam of a drawing reads the same document, encoded once for all of them and every slot.
  val encoded = encode(stack(documents.flatMap(document => Seq.fill(width)(document)), carried.axis))

  /** What the queries answer for every partial record, read back to the host once per step — a
    * slice at a time would be thousands of transfers for one step.
    */
  def asked(): (NodeAnswers, EdgeAnswers) =
    val laid = beams.flatMap(drawing => drawing.padTo(width, drawing.last).map(_.record))
    val said = ask(encoded, RecordBatch.of(laid, carried.axis, nodes, edges))
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
