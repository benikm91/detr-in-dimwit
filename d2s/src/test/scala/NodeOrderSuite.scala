import d2s.*
import d2s.sketch.NodeOrder
import dataset.NodeClass
import dataset.Point
import dataset.RecordBatch
import dataset.RecordGraph
import dataset.RecordNode
import dimwit.*

class NodeOrderSuite extends munit.FunSuite:

  override def beforeAll(): Unit = dimwit.initialize()

  private trait Drawing derives Label
  private trait Edge derives Label

  private val classesWritten = Seq(
    "simplest-first" -> Seq(NodeClass.Point, NodeClass.Line, NodeClass.Line, NodeClass.Circle, NodeClass.Arc),
    "simplest-last" -> Seq(NodeClass.Arc, NodeClass.Circle, NodeClass.Line, NodeClass.Line, NodeClass.Point)
  )

  for (order, classes) <- classesWritten do
    test(s"a record laid out $order is written class by class, in any order within a class"):
      val sketch = RecordGraph(
        Seq(
          RecordNode(NodeClass.Arc, false, Seq(Point(0.2f, 0.5f), Point(0.6f, 0.5f), Point(0.4f, 0.3f))),
          RecordNode(NodeClass.Line, false, Seq(Point(0.1f, 0.1f), Point(0.9f, 0.1f))),
          RecordNode(NodeClass.Point, false, Seq(Point(0.5f, 0.5f))),
          RecordNode(NodeClass.Circle, true, Seq(Point(0.3f, 0.7f), Point(0.5f, 0.7f))),
          RecordNode(NodeClass.Line, false, Seq(Point(0.1f, 0.9f), Point(0.9f, 0.9f)))
        ),
        Seq.empty
      )
      val (nodes, edges) = (Axis[Node] -> 8, Axis[Edge] -> 0)
      val laid = RecordBatch.of(Seq(sketch), Axis[Drawing], nodes, edges)
      val orders = (1 to 10).map: seed =>
        val written = RecordGraph.of(laid.permuted(Random.Key(seed), nodes, edges, NodeOrder.named(order).classRank)).head.nodes
        assertEquals(written.toSet, sketch.nodes.toSet, s"seed $seed")
        assertEquals(written.map(_.nodeClass), classes, s"seed $seed")
        written
      assert(orders.distinct.size > 1, "the two lines come in either order")
