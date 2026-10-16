package syntheticFloorplansDataset

import dataset.DrawingDataset.Split
import dimwit.*
import munit.FunSuite

/** The synthetic floor plans, read from Hugging Face: what the records hold, and how the stream runs. */
class SyntheticFloorplansSuite extends FunSuite:

  override def beforeAll(): Unit = dimwit.initialize()

  /** The validation split is downloaded the first time it is opened. */
  override def munitTimeout: scala.concurrent.duration.Duration = scala.concurrent.duration.Duration(20, "min")

  private trait Width derives Label
  private trait Height derives Label
  private trait Channel derives Label
  private trait Node derives Label
  private trait Edge derives Label
  private trait Batch derives Label

  private lazy val validation = SyntheticFloorplans.open(Axis[Width], Axis[Height], Axis[Channel], Axis[Node], Axis[Edge])(Split.Validation)

  test("the training split is its nine parts in seed order, and validation one part"):
    val train = SyntheticFloorplans.open(Axis[Width], Axis[Height], Axis[Channel], Axis[Node], Axis[Edge])(Split.Train)
    assertEquals(train.parts, (0 until 9).map(part => (s"train-0$part", 50_000)))
    assertEquals(validation.parts, Seq(("val", 5000)))

  test("a record holds horizontal and vertical segments, openings in the wall they belong in, walls joined at a shared end"):
    for sample <- validation.samples.take(40) do
      val (record, related) = (sample.target.nodes, sample.target.edges)
      val classes = record.nodeClass.toArray.map(FloorplanClass.fromId)
      val drawn = classes.indices.filter(classes(_).isNode)
      assert(drawn.nonEmpty && drawn == classes.indices.take(drawn.size), "the nodes come first, the padding after")
      val (startX, startY, endX, endY) = (record.startX.toArray, record.startY.toArray, record.endX.toArray, record.endY.toArray)
      def ends(at: Int) = Set((startX(at), startY(at)), (endX(at), endY(at)))
      for at <- drawn do
        assert(Seq(startX(at), startY(at), endX(at), endY(at)).forall(v => v >= 0f && v <= 1f), s"node $at lies off the drawing")
        assert(startX(at) == endX(at) || startY(at) == endY(at), s"node $at is not horizontal or vertical")
        assert(startX(at) <= endX(at) && startY(at) <= endY(at), s"node $at is not read rightwards or downwards")
      val (relationship, subject, obj) = (related.relationship.toArray.map(FloorplanRelationship.fromId), related.subject.toArray, related.obj.toArray)
      val walls = Set(FloorplanClass.OutsideWall, FloorplanClass.InsideWall)
      for at <- relationship.indices if relationship(at).isRelationship do
        relationship(at) match
          case FloorplanRelationship.Joined =>
            assert(walls(classes(subject(at))) && walls(classes(obj(at))) && subject(at) < obj(at))
            assert((ends(subject(at)) & ends(obj(at))).nonEmpty, s"walls ${subject(at)} and ${obj(at)} share no end")
          case FloorplanRelationship.In =>
            val belongs = if classes(subject(at)) == FloorplanClass.Door then FloorplanClass.InsideWall else FloorplanClass.OutsideWall
            assertEquals(classes(obj(at)), belongs, s"a ${classes(subject(at))} in a ${classes(obj(at))}")
          case FloorplanRelationship.NoRelationship => ()
      val hosted = relationship.indices.filter(relationship(_) == FloorplanRelationship.In).map(subject).sorted
      assertEquals(hosted, drawn.filterNot(at => walls(classes(at))).sorted, "every opening stands in exactly one wall")

  test("the batches carry on where they left off, and wrap round at the end of the split"):
    val batch = Axis[Batch] -> 8
    def first(afterSteps: Int) = validation.batches(batch, afterSteps).next()
    val stream = validation.batches(batch, afterSteps = 0)
    val fourth = stream.drop(3).next()
    assertEquals(first(3).target.nodeClass.toArray.toSeq.map(_.toSeq), fourth.target.nodeClass.toArray.toSeq.map(_.toSeq))
    // 5,000 plans make 625 whole batches of 8: the 626th is the first again.
    assertEquals(first(625).target.startX.toArray.toSeq.map(_.toSeq), first(0).target.startX.toArray.toSeq.map(_.toSeq))
    assertEquals(fourth.pixels.shape(Axis[Width]), SyntheticFloorplans.canvas)

  test("a record laid out again keeps its nodes, and its relationships between the same nodes"):
    trait Plan derives Label
    def node(nodeClass: FloorplanClass, x0: Float, y0: Float, x1: Float, y1: Float) = FloorplanNode(nodeClass, dataset.Point(x0, y0), dataset.Point(x1, y1))
    val plan = FloorplanGraph(
      Seq(
        node(FloorplanClass.OutsideWall, 0.1f, 0.1f, 0.9f, 0.1f),
        node(FloorplanClass.Window, 0.3f, 0.1f, 0.5f, 0.1f),
        node(FloorplanClass.OutsideWall, 0.9f, 0.1f, 0.9f, 0.9f),
        node(FloorplanClass.InsideWall, 0.5f, 0.1f, 0.5f, 0.9f),
        node(FloorplanClass.Door, 0.5f, 0.4f, 0.5f, 0.5f)
      ),
      Seq(
        FloorplanEdge(FloorplanRelationship.Joined, 0, 2),
        FloorplanEdge(FloorplanRelationship.In, 1, 0),
        FloorplanEdge(FloorplanRelationship.In, 4, 3)
      )
    )
    val (nodes, edges) = (Axis[Node] -> 8, Axis[Edge] -> 5)
    val laid = FloorplanRecordBatch.of(Seq(plan), Axis[Plan], nodes, edges)
    // A symmetric relationship says nothing by which of its two nodes comes first.
    def related(graph: FloorplanGraph) = graph.edges.map: edge =>
      val (from, to) = (graph.nodes(edge.subject), graph.nodes(edge.obj))
      (edge.relationship, if edge.relationship.isSymmetric then Set(from, to) else Set(from), Set(from, to))
    .toSet
    val wallsFirst = Tensor1(Axis[FloorplanClasses], VType[Float32]).fromArray(FloorplanClass.values.map(c => if c.isNode then (if c == FloorplanClass.OutsideWall || c == FloorplanClass.InsideWall then 0f else 1f) else 2f))
    val orders = (1 to 8).map: seed =>
      val written = FloorplanGraph.of(laid.permuted(Random.Key(seed), nodes, edges, wallsFirst)).head
      assertEquals(written.nodes.toSet, plan.nodes.toSet, s"seed $seed")
      assertEquals(related(written), related(plan), s"seed $seed: the relationships no longer link the same nodes")
      assert(written.nodes.take(3).forall(n => n.nodeClass == FloorplanClass.OutsideWall || n.nodeClass == FloorplanClass.InsideWall), s"seed $seed: walls first")
      written.nodes
    assert(orders.distinct.size > 1, "a fresh order every time")

  test("a plan's record drawn on an empty canvas colours the pixels its nodes run through, and no others"):
    val plan = FloorplanGraph.of(validation.samples.next().target)
    val canvas = Tensor(Shape2(Axis[Width] -> SyntheticFloorplans.canvas, Axis[Height] -> SyntheticFloorplans.canvas), VType[UInt8]).fill(255)
    val drawn = FloorplanDrawing(plan, canvas, Axis[Channel])
    val inked = drawn.slice(Axis[Channel].at(0)).asInt(VType[Int32]).toArray.map(_.map(_ != 255))
    def at(point: dataset.Point) = ((point.x * SyntheticFloorplans.canvas).round.min(SyntheticFloorplans.canvas - 1), (point.y * SyntheticFloorplans.canvas).round.min(SyntheticFloorplans.canvas - 1))
    for node <- plan.nodes; (x, y) <- Seq(at(node.start), at(node.end)) do
      assert(inked(x)(y), s"$node is not drawn at its end ($x, $y)")
    assert(inked.flatten.count(identity) < inked.flatten.length / 4, "the record covers a quarter of the canvas or more")
