import d2tok.*
import d2tok.floorplan.*
import dataset.Point
import dimwit.*
import syntheticFloorplansDataset.FloorplanClass
import syntheticFloorplansDataset.FloorplanGraph
import syntheticFloorplansDataset.FloorplanNode
import syntheticFloorplansDataset.FloorplanRecordBatch

class FloorplanWordsSuite extends munit.FunSuite:

  override def beforeAll(): Unit = dimwit.initialize()

  private trait Drawing derives Label

  private val canvas = 64
  private val words = Words(canvas)
  private def at(x: Int, y: Int) = Point(x.toFloat / canvas, y.toFloat / canvas)
  private val plan = FloorplanGraph(
    Seq(
      FloorplanNode(FloorplanClass.Window, at(4, 20), at(4, 30)),
      FloorplanNode(FloorplanClass.OutsideWall, at(4, 4), at(60, 4)),
      FloorplanNode(FloorplanClass.InsideWall, at(30, 4), at(30, 60)),
      FloorplanNode(FloorplanClass.OutsideWall, at(4, 4), at(4, 60)),
      FloorplanNode(FloorplanClass.Door, at(30, 10), at(30, 18))
    ),
    Seq.empty
  )
  private val (nodes, relationships) = (Axis[Node] -> 6, Axis[Relationship] -> 0)
  private lazy val tokens = Axis[Token] -> words.tokens(nodes.size)

  test("a plan written in words reads back as the same nodes, followed by its end"):
    val (written, count) = words.written(plan.record(nodes, relationships).nodes, tokens)
    val said = written.toArray.toSeq
    assertEquals(count.item, 5 * 5 + 1, "five nodes of five words, then the end")
    assertEquals(said.drop(count.item - 1).distinct, Seq(words.end), "the end, and nothing but the end after it")
    assertEquals(words.read(said), plan.nodes)
    assertEquals(words.read(said.updated(3, words.end)), Seq.empty, "a node cut short by the end")

  /** The plan laid out in an order. */
  private def laidOut(order: SequenceOrder, seed: Int) =
    val records = FloorplanRecordBatch.of(Seq(plan), Axis[Drawing], nodes, relationships)
    FloorplanGraph.of(order.laidOut(records, Random.Key(seed), canvas)).head.nodes

  test("by position, a plan is written top to bottom and left to right, by class where two start on the same pixel"):
    assertEquals(laidOut(SequenceOrder.ByPosition, 1), Seq(plan.nodes(1), plan.nodes(3), plan.nodes(2), plan.nodes(4), plan.nodes(0)))

  test("by class, a plan is written walls first, in any order within a class"):
    val orders = (1 to 10).map: seed =>
      val written = laidOut(SequenceOrder.ByClass, seed)
      assertEquals(written.map(_.nodeClass), Seq(FloorplanClass.OutsideWall, FloorplanClass.OutsideWall, FloorplanClass.InsideWall, FloorplanClass.Door, FloorplanClass.Window), s"seed $seed")
      written
    assert(orders.distinct.size > 1, "the two outside walls come in either order")
