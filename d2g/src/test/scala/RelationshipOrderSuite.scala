import d2g.*
import d2g.floorplan.RelationshipOrder
import d2s.floorplan.NodeOrder
import dimwit.*
import syntheticFloorplansDataset.FloorplanClass
import syntheticFloorplansDataset.FloorplanEdge
import syntheticFloorplansDataset.FloorplanGraph
import syntheticFloorplansDataset.FloorplanNode
import syntheticFloorplansDataset.FloorplanRecordBatch
import syntheticFloorplansDataset.FloorplanRelationship

class RelationshipOrderSuite extends munit.FunSuite:

  override def beforeAll(): Unit = dimwit.initialize()

  private trait Plan derives Label

  test("a record laid out by kind writes the joined walls, then the windows in theirs, then the doors, in any order within a kind"):
    def node(nodeClass: FloorplanClass, x0: Float, y0: Float, x1: Float, y1: Float) = FloorplanNode(nodeClass, dataset.Point(x0, y0), dataset.Point(x1, y1))
    val plan = FloorplanGraph(
      Seq(
        node(FloorplanClass.Door, 0.5f, 0.4f, 0.5f, 0.5f),
        node(FloorplanClass.OutsideWall, 0.1f, 0.1f, 0.9f, 0.1f),
        node(FloorplanClass.Window, 0.3f, 0.1f, 0.5f, 0.1f),
        node(FloorplanClass.OutsideWall, 0.9f, 0.1f, 0.9f, 0.9f),
        node(FloorplanClass.InsideWall, 0.5f, 0.1f, 0.5f, 0.9f),
        node(FloorplanClass.OutsideDoor, 0.9f, 0.4f, 0.9f, 0.5f)
      ),
      Seq(
        FloorplanEdge(FloorplanRelationship.In, 0, 4),
        FloorplanEdge(FloorplanRelationship.In, 2, 1),
        FloorplanEdge(FloorplanRelationship.Joined, 1, 3),
        FloorplanEdge(FloorplanRelationship.In, 5, 3),
        FloorplanEdge(FloorplanRelationship.Joined, 1, 4)
      )
    )
    val (nodes, edges) = (Axis[Node] -> 8, Axis[Edge] -> 7)
    val laid = FloorplanRecordBatch.of(Seq(plan), Axis[Plan], nodes, edges)
    def kind(graph: FloorplanGraph, edge: FloorplanEdge) = graph.nodes(edge.subject).nodeClass match
      case FloorplanClass.Window                           => "window"
      case FloorplanClass.Door | FloorplanClass.OutsideDoor => "door"
      case _                                               => "joined"
    val orders = (1 to 10).map: seed =>
      val written = FloorplanGraph.of(laid.permuted(Random.Key(seed), nodes, edges, NodeOrder.WallsFirst.classRank, RelationshipOrder.ByKind.startRank)).head
      assertEquals(written.edges.map(kind(written, _)), Seq("joined", "joined", "window", "door", "door"), s"seed $seed")
      written.edges.map(edge => (written.nodes(edge.subject), written.nodes(edge.obj)))
    assert(orders.distinct.size > 1, "the relationships of a kind come in either order")
