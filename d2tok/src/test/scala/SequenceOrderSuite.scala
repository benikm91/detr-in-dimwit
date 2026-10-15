import d2tok.*
import dataset.NodeClass
import dataset.Point
import dataset.RecordBatch
import dataset.RecordGraph
import dataset.RecordNode
import dimwit.*

class SequenceOrderSuite extends munit.FunSuite:

  override def beforeAll(): Unit = dimwit.initialize()

  private trait Drawing derives Label

  private val canvas = 64
  private def at(x: Int, y: Int) = Point(x.toFloat / canvas, y.toFloat / canvas)
  private val corner = at(10, 10)
  private val sketch = Seq(
    RecordNode(NodeClass.Arc, false, Seq(at(30, 40), at(50, 40), at(40, 30))),
    RecordNode(NodeClass.Line, false, Seq(corner, at(10, 50))),
    RecordNode(NodeClass.Point, false, Seq(at(5, 20))),
    RecordNode(NodeClass.Line, false, Seq(corner, at(50, 10))),
    RecordNode(NodeClass.Circle, true, Seq(at(20, 30), at(28, 30)))
  )
  private val (nodes, relationships) = (Axis[Node] -> 6, Axis[Relationship] -> 0)

  /** The sketch laid out in an order, from its nodes in the order `listed`. */
  private def laidOut(order: SequenceOrder, listed: Seq[RecordNode], seed: Int) =
    val records = RecordBatch.of(Seq(RecordGraph(listed, Seq.empty)), Axis[Drawing], nodes, relationships)
    RecordGraph.of(order.laidOut(records, Random.Key(seed), canvas)).head.nodes

  test("by position, a record is written top to bottom and left to right, whatever order it is listed in"):
    // The two lines start on the same corner: the one ending to its right comes first, as it ends higher up.
    val expected = Seq(sketch(3), sketch(1), sketch(2), sketch(4), sketch(0))
    for seed <- 1 to 5 do
      val listed = scala.util.Random(seed).shuffle(sketch)
      assertEquals(laidOut(SequenceOrder.ByPosition, listed, seed), expected, s"listed ${listed.map(_.nodeClass)}")

  test("by class, a record is written class by class, in any order within a class"):
    val orders = (1 to 10).map: seed =>
      val written = laidOut(SequenceOrder.ByClass, sketch, seed)
      assertEquals(written.toSet, sketch.toSet, s"seed $seed")
      assertEquals(written.map(_.nodeClass), Seq(NodeClass.Point, NodeClass.Line, NodeClass.Line, NodeClass.Circle, NodeClass.Arc), s"seed $seed")
      written
    assert(orders.distinct.size > 1, "the two lines come in either order")
