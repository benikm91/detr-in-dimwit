import d2tok.*
import d2tok.floorplan.SequenceOrder
import d2tok.floorplangraph.RelationshipOrder
import d2tok.floorplangraph.config.D2TokGraphSetup
import dataset.Point
import dimwit.*
import syntheticFloorplansDataset.FloorplanClass
import syntheticFloorplansDataset.FloorplanCorpus
import syntheticFloorplansDataset.FloorplanEdge
import syntheticFloorplansDataset.FloorplanEdges
import syntheticFloorplansDataset.FloorplanGraph
import syntheticFloorplansDataset.FloorplanNode
import syntheticFloorplansDataset.FloorplanNodes
import syntheticFloorplansDataset.FloorplanRecord
import syntheticFloorplansDataset.FloorplanRecordBatch
import syntheticFloorplansDataset.FloorplanRelationship

/** A plan's relationships name the right nodes once its nodes are permuted, numbered and written. */
class FloorplanGraphWordsSuite extends munit.FunSuite:

  override def beforeAll(): Unit = dimwit.initialize()

  private trait Drawing derives Label

  private val setup = D2TokGraphSetup(FloorplanCorpus.Furnished, checkpointRoot = "unused")
  private val canvas = setup.canvas
  private val (nodes, relationships) = (Axis[Node] -> setup.corpus.maxNodes, Axis[Relationship] -> setup.corpus.maxRelationships)
  private val tokens = Axis[Token] -> setup.tokens

  private def at(x: Int, y: Int) = Point(x.toFloat / canvas, y.toFloat / canvas)
  private val nowhere = at(0, 0)

  // Two rooms of a corner, its walls, a door and a window in them and a bed, listed out of order.
  private val plan = FloorplanGraph(
    nodes = Seq(
      FloorplanNode(FloorplanClass.Window, at(4, 20), at(4, 30)),
      FloorplanNode(FloorplanClass.OutsideWall, at(4, 4), at(60, 4)),
      FloorplanNode(FloorplanClass.InsideWall, at(30, 4), at(30, 60)),
      FloorplanNode(FloorplanClass.Living, nowhere, nowhere),
      FloorplanNode(FloorplanClass.OutsideWall, at(4, 4), at(4, 60)),
      FloorplanNode(FloorplanClass.Door, at(30, 10), at(30, 18)),
      FloorplanNode(FloorplanClass.Bed, at(40, 40), at(50, 55)),
      FloorplanNode(FloorplanClass.Kitchen, nowhere, nowhere)
    ),
    edges = Seq(
      FloorplanEdge(FloorplanRelationship.Borders, 2, 3),
      FloorplanEdge(FloorplanRelationship.Joined, 1, 2),
      FloorplanEdge(FloorplanRelationship.In, 5, 2),
      FloorplanEdge(FloorplanRelationship.Joined, 4, 1),
      FloorplanEdge(FloorplanRelationship.In, 6, 3),
      FloorplanEdge(FloorplanRelationship.In, 0, 4),
      FloorplanEdge(FloorplanRelationship.Borders, 4, 7)
    )
  )

  /** The relationships by the nodes they name rather than by number, so that two numberings of the
    * same record compare equal.
    */
  private def related(record: FloorplanGraph): Set[(FloorplanRelationship, Set[FloorplanNode], Seq[FloorplanNode])] =
    record.edges.map: edge =>
      val ends = Seq(record.nodes(edge.subject), record.nodes(edge.obj))
      if edge.relationship.isSymmetric then (edge.relationship, ends.toSet, Seq.empty) else (edge.relationship, Set.empty[FloorplanNode], ends)
    .toSet

  /** The plan laid out as training lays it out, written, and read back. */
  private def writtenAndRead(written: D2TokGraphSetup, seed: Int): (Seq[Int], FloorplanGraph) =
    val laid = FloorplanRecordBatch.of(Seq(plan), Axis[Drawing], nodes, relationships)
      .permuted(Random.Key(seed), nodes, relationships, written.nodeClassRank, written.relationshipOrder.kindRank)
    val words = zipvmap(Axis[Drawing])(laid.nodeClass, laid.startX, laid.startY, laid.endX, laid.endY, laid.relationship, laid.subject, laid.obj):
      case (nodeClass, startX, startY, endX, endY, relationship, subject, obj) =>
        written.words.written(FloorplanRecord(FloorplanNodes(nodeClass, startX, startY, endX, endY), FloorplanEdges(relationship, subject, obj)), tokens)._1
    val said = words.toArray.head.toSeq
    (said, written.words.read(said))

  for
    order <- Seq(SequenceOrder.Random, SequenceOrder.WallsFirst)
    relationshipOrder <- Seq(RelationshipOrder.Free, RelationshipOrder.ByKind)
  do
    test(s"written in order ${order.name}, relationships $relationshipOrder, a plan reads back as itself"):
      val written = setup.copy(nodeOrder = order, relationshipOrder = relationshipOrder)
      val orders = (1 to 20).map: seed =>
        val (_, read) = writtenAndRead(written, seed)
        assertEquals(read.nodes.size, plan.nodes.size, s"seed $seed")
        assertEquals(read.nodes.toSet, plan.nodes.toSet, s"seed $seed")
        assertEquals(related(read), related(plan), s"seed $seed")
        if order == SequenceOrder.WallsFirst then
          import FloorplanClass.*
          assertEquals(read.nodes.map(_.nodeClass), Seq(OutsideWall, OutsideWall, InsideWall, Door, Window, Living, Kitchen, Bed), s"seed $seed")
        if relationshipOrder == RelationshipOrder.ByKind then
          import FloorplanRelationship.*
          assertEquals(read.edges.map(_.relationship), Seq(Joined, Joined, In, In, In, Borders, Borders), s"seed $seed")
          assertEquals(read.nodes(read.edges(2).subject).nodeClass, FloorplanClass.Window, s"seed $seed: windows come first among what is in something")
        read.nodes
      assert(orders.distinct.size > 1, "the nodes are permuted")

  test("every node writes its number first, counting up from 0, and nothing follows the end"):
    val (said, read) = writtenAndRead(setup, seed = 3)
    val firstNumber = setup.words.size - setup.corpus.maxNodes
    val numbersAtNodes = said.grouped(setup.words.perNode).take(read.nodes.size).map(_.head - firstNumber).toSeq
    assertEquals(numbersAtNodes, read.nodes.indices)
    val wordsUsed = read.nodes.size * setup.words.perNode + read.edges.size * setup.words.perRelationship
    assert(said.drop(wordsUsed).forall(_ == setup.words.end), "the words after the record are its end")
