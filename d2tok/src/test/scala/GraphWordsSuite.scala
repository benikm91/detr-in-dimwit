import d2tok.*
import d2tok.sketch.SequenceOrder
import d2tok.sketchgraph.RelationshipOrder
import d2tok.sketchgraph.config.D2TokGraphSetup
import dataset.Corpus
import dataset.EdgeClass
import dataset.NodeClass
import dataset.Point
import dataset.Record
import dataset.RecordBatch
import dataset.RecordEdge
import dataset.RecordGraph
import dataset.RecordNode
import dimwit.*

/** A drawing's relationships name the right nodes once its nodes are permuted, numbered and written. */
class GraphWordsSuite extends munit.FunSuite:

  override def beforeAll(): Unit = dimwit.initialize()

  private trait Drawing derives Label

  private val setup = D2TokGraphSetup(Corpus.Rectilinear6to18, checkpointRoot = "unused")
  private val canvas = setup.corpus.canvas
  private val (nodes, relationships) = (Axis[Node] -> setup.corpus.maxNodes, Axis[Relationship] -> setup.corpus.maxEdges)
  private val tokens = Axis[Token] -> setup.tokens

  private def at(x: Int, y: Int) = Point(x.toFloat / canvas, y.toFloat / canvas)
  private def line(from: Point, to: Point) = RecordNode(NodeClass.Line, false, Seq(from, to))
  private def annotation(where: Point) = RecordNode(NodeClass.Annotation, false, Seq(where))

  // A rectangle and two of its sides annotated, listed with the annotations among the lines.
  private val (topLeft, topRight, bottomLeft, bottomRight) = (at(40, 40), at(200, 40), at(40, 180), at(200, 180))
  private val drawing = RecordGraph(
    nodes = Seq(
      annotation(at(120, 30)),
      line(topLeft, topRight),
      line(topLeft, bottomLeft),
      annotation(at(210, 110)),
      line(topRight, bottomRight),
      line(bottomLeft, bottomRight)
    ),
    edges = Seq(
      RecordEdge(EdgeClass.Connected, 1, 2),
      RecordEdge(EdgeClass.Connected, 1, 4),
      RecordEdge(EdgeClass.Connected, 2, 5),
      RecordEdge(EdgeClass.Connected, 4, 5),
      RecordEdge(EdgeClass.Annotates, 0, 1),
      RecordEdge(EdgeClass.Annotates, 3, 4)
    )
  )

  /** The relationships by the nodes they name rather than by number, so that two numberings of the
    * same record compare equal.
    */
  private def related(record: RecordGraph): Set[(EdgeClass, Set[RecordNode], Seq[RecordNode])] =
    record.edges.map: edge =>
      val ends = Seq(record.nodes(edge.subject), record.nodes(edge.obj))
      if edge.edgeClass.isSymmetric then (edge.edgeClass, ends.toSet, Seq.empty) else (edge.edgeClass, Set.empty[RecordNode], ends)
    .toSet

  /** The drawing laid out as training lays it out, written, and read back. */
  private def writtenAndRead(written: D2TokGraphSetup, seed: Int): (Seq[Int], RecordGraph) =
    val laid = RecordBatch.of(Seq(drawing), Axis[Drawing], nodes, relationships)
      .permuted(Random.Key(seed), nodes, relationships, written.nodeClassRank, written.relationshipOrder.classRank)
    val words = zipvmap(Axis[Drawing])(laid.nodeClass, laid.construction, laid.startX, laid.startY, laid.endX, laid.endY, laid.midX, laid.midY, laid.edgeClass, laid.subject, laid.obj):
      case (nodeClass, construction, startX, startY, endX, endY, midX, midY, edgeClass, subject, obj) =>
        written.words.written(Record(nodeClass, construction, startX, startY, endX, endY, midX, midY, edgeClass, subject, obj), tokens)._1
    val said = words.toArray.head.toSeq
    (said, written.words.read(said))

  for
    order <- Seq(SequenceOrder.Random, SequenceOrder.ByClass)
    relationshipOrder <- Seq(RelationshipOrder.Free, RelationshipOrder.ByKind)
  do
    test(s"written in order ${order.name}, relationships ${relationshipOrder}, a drawing reads back as itself"):
      val written = setup.copy(order = order, relationshipOrder = relationshipOrder)
      val orders = (1 to 20).map: seed =>
        val (_, read) = writtenAndRead(written, seed)
        assertEquals(read.nodes.size, drawing.nodes.size, s"seed $seed")
        assertEquals(read.nodes.toSet, drawing.nodes.toSet, s"seed $seed")
        assertEquals(related(read), related(drawing), s"seed $seed")
        if order == SequenceOrder.ByClass then
          assertEquals(read.nodes.map(_.nodeClass), Seq.fill(4)(NodeClass.Line) ++ Seq.fill(2)(NodeClass.Annotation), s"seed $seed")
        if relationshipOrder == RelationshipOrder.ByKind then
          assertEquals(read.edges.map(_.edgeClass), Seq.fill(4)(EdgeClass.Connected) ++ Seq.fill(2)(EdgeClass.Annotates), s"seed $seed")
        read.nodes
      assert(orders.distinct.size > 1, "the nodes are permuted")

  test("every node writes its number first, counting up from 0, and nothing follows the end"):
    val (said, read) = writtenAndRead(setup, seed = 3)
    val firstNumber = setup.words.size - setup.corpus.maxNodes
    val numbersAtNodes = Iterator.unfold((said, 0)):
      case (number +: rest, next) if next < read.nodes.size => Some((number - firstNumber, (rest.drop(1 + 2 * read.nodes(next).nodeClass.numPoints), next + 1)))
      case _ => None
    .toSeq
    assertEquals(numbersAtNodes, read.nodes.indices)
    val wordsUsed = read.nodes.map(node => 2 + 2 * node.nodeClass.numPoints).sum + read.edges.size * setup.words.perRelationship
    assert(said.drop(wordsUsed).forall(_ == setup.words.end), "the words after the record are its end")
