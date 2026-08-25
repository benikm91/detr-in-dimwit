package dataset

import dataset.LShapeDataset.Split
import dimwit.*
import munit.FunSuite

/** The two views of a drawing, and that neither loses what the other holds. */
class ObjectsSuite extends FunSuite:

  override def beforeAll(): Unit = dimwit.initialize()

  /** Both splits are parsed the first time they are opened. */
  override def munitTimeout: scala.concurrent.duration.Duration = scala.concurrent.duration.Duration(10, "min")

  private trait Width derives Label
  private trait Height derives Label
  private trait Channel derives Label
  private trait Node derives Label
  private trait Edge derives Label
  private trait Drawing derives Label

  private val nodes = Axis[Node] -> 8
  private val edges = Axis[Edge] -> 4

  private val record = RecordGraph(
    nodes = Seq(
      RecordNode(NodeClass.Line, Seq(Point(0.2f, 0.4f), Point(0.8f, 0.4f))),
      RecordNode(NodeClass.Line, Seq(Point(0.8f, 0.4f), Point(0.8f, 0.9f))),
      RecordNode(NodeClass.Annotation, Seq(Point(0.5f, 0.3f)))
    ),
    edges = Seq(
      RecordEdge(EdgeClass.Connected, 0, 1),
      RecordEdge(EdgeClass.Annotates, 2, 0)
    )
  )

  test("a line is drawn as the box between its end points, an annotation as a box around it"):
    val objects = Objects.of(record.record(nodes, edges))
    val box = objects.detection.box
    assertEquals(objects.detection.label.toArray.toSeq, Seq(1, 1, 2, 0, 0, 0, 0, 0))
    assertEqualsFloat(box.centerX.toArray(0), 0.5f, 1e-6f)
    assertEqualsFloat(box.width.toArray(0), 0.6f, 1e-6f)
    assertEqualsFloat(box.height.toArray(0), 4f / Canvas, 1e-6f)
    assertEqualsFloat(box.width.toArray(2), 12f / Canvas, 1e-6f)
    assert(box.width.toArray.drop(3).forall(_ == 0f), "a relationship is not drawn")

  test("a symmetric relationship is drawn both ways round, a directed one is not"):
    val relations = Objects.of(record.record(nodes, edges)).relations.toArray
    assertEquals(relations(0)(1)(RelationClass.Connected.id), 1f)
    assertEquals(relations(1)(0)(RelationClass.Connected.id), 1f)
    assertEquals(relations(2)(0)(RelationClass.Annotates.id), 1f)
    assertEquals(relations(0)(2)(RelationClass.Annotates.id), 0f)
    assertEquals(relations.flatten.flatten.sum, 3f)

  test("a record survives being permuted, laid out and read back"):
    val random = scala.util.Random(7)
    for _ <- 1 to 20 do
      val permuted = RecordGraph.of(record.permuted(random).record(nodes, edges))
      assertEquals(permuted.nodes.toSet, record.nodes.toSet)
      assertEquals(related(permuted), related(record))

  test("a record permuted on the device is the same record, laid out again"):
    val laid = RecordBatch.of(Seq(record, record), Axis[Drawing], nodes, edges)
    val (nodeSlots, edgeSlots) = (Axis[Node] -> 10, Axis[Edge] -> 6)
    val compiled = jit: (records: RecordBatch[Drawing, Node, Edge], key: dimwit.Random.Key) =>
      records.permuted(key, nodeSlots, edgeSlots)
    for seed <- 1 to 10 do
      val permuted = laid.permuted(dimwit.Random.Key(seed), nodeSlots, edgeSlots)
      // A key is an argument of the compiled permutation, not something baked into it.
      assertEquals(read(compiled(laid, dimwit.Random.Key(seed))), read(permuted), s"seed $seed")
      assertEquals(permuted.nodeClass.shape(Axis[Node]), nodeSlots.size)
      assertEquals(permuted.edgeClass.shape(Axis[Edge]), edgeSlots.size)
      RecordGraph.of(permuted).zipWithIndex.foreach: (found, drawing) =>
        assertEquals(found.nodes.toSet, record.nodes.toSet, s"seed $seed, drawing $drawing")
        assertEquals(related(found), related(record), s"seed $seed, drawing $drawing")
      permuted.nodeClass.toArray.zipWithIndex.foreach: (drawing, at) =>
        val classes = drawing.map(NodeClass.fromId).toSeq
        assertEquals(classes.sortBy(held => if held.isDrawn then 0 else 1), classes, s"seed $seed, drawing $at: nodes before empty positions")
      permuted.edgeClass.toArray.zipWithIndex.foreach: (drawing, at) =>
        val classes = drawing.map(EdgeClass.fromId).toSeq
        assertEquals(classes.sortBy(held => if held.relates then 0 else 1), classes, s"seed $seed, drawing $at: relationships before empty positions")
      permuted.edgeClass.toArray.zip(permuted.links.toArray).foreach: (drawing, links) =>
        drawing.zip(links).foreach: (held, ends) =>
          val edgeClass = EdgeClass.fromId(held)
          if edgeClass.isSymmetric then assert(ends(0) < ends(1), s"$edgeClass holds ${ends.toSeq} rather than its ends in ascending order")
          if !edgeClass.relates then assertEquals(ends.toSeq, Seq(0, 0), s"an empty position links nothing")

  test("a record is drawn as the picture it stands for"):
    val canvas = 32
    val blank = Outlines.greyLevels(Tensor3(Axis[Width] -> canvas, Axis[Height] -> canvas, Axis[Channel] -> 1, VType[Float32]).fill(1f))
    val drawn = RecordDrawing(record, blank, Axis[Channel]).asInt(VType[Int32]).toArray
    assertEquals(drawn.length, canvas)
    assertEquals(drawn.head.head.toSeq, Seq(255, 255, 255), "a corner the record does not reach stays blank")
    // The record's first line runs from (0.2, 0.4) to (0.8, 0.4), so the middle of the canvas is on it.
    assertEquals(drawn(canvas / 2)(math.round(0.4f * canvas)).toSeq, Seq(20, 60, 190), "the middle of a line is drawn in the line's colour")

  test("a record is drawn where the drawing it was read from has its ink"):
    val data = LShapeDataset.open(Axis[Width], Axis[Height], Axis[Channel], Axis[Node], Axis[Edge])(Split.Validation)
    data.samples.take(3).zipWithIndex.foreach: (sample, index) =>
      val drawing = Outlines.greyLevels(sample.image)
      val drawn = RecordDrawing(RecordGraph.of(sample.target), drawing, Axis[Channel]).asInt(VType[Int32]).toArray
      val ink = drawing.asInt(VType[Int32]).toArray
      def inked(x: Int, y: Int) = ink.isDefinedAt(x) && ink(x).isDefinedAt(y) && ink(x)(y) < 128
      val onLine =
        for
          x <- drawn.indices
          y <- drawn(x).indices
          if drawn(x)(y).toSeq == Seq(20, 60, 190)
        yield Seq(-1, 0, 1).exists(dx => Seq(-1, 0, 1).exists(dy => inked(x + dx, y + dy)))
      assert(onLine.nonEmpty, s"sample $index: a record of lines drew none")
      assertEquals(onLine.count(identity), onLine.size, s"sample $index: a line is drawn where the drawing has no ink near it")

  test("the objects of a record hold the record"):
    assertSameRecord(Objects.record(Objects.of(record.record(nodes, edges)), edges), record.record(nodes, edges))

  Split.values.foreach: split =>
    test(s"the objects of every record of the ${split.fileName} split hold that record"):
      val data = LShapeDataset.open(Axis[Width], Axis[Height], Axis[Channel], Axis[Node], Axis[Edge])(split)
      val edgeSlots = Axis[Edge] -> MaxEdges
      data.samples.take(200).zipWithIndex.foreach: (sample, index) =>
        assertSameRecord(Objects.record(Objects.of(sample.target), edgeSlots), sample.target, s"sample $index")

  /** What a record's relationships say in terms of its nodes rather than of their slots. A
    * symmetric relationship says nothing by which of its two nodes comes first.
    */
  private def related(record: RecordGraph): Set[(EdgeClass, Set[RecordNode], Seq[RecordNode])] =
    record.edges.map: edge =>
      val ends = Seq(record.nodes(edge.subject), record.nodes(edge.obj))
      if edge.edgeClass.isSymmetric then (edge.edgeClass, ends.toSet, Seq.empty) else (edge.edgeClass, Set.empty[RecordNode], ends)
    .toSet

  private def assertSameRecord(actual: Record[Node, Edge], expected: Record[Node, Edge], clue: String = ""): Unit =
    assertEquals(actual.nodeClass.toArray.toSeq, expected.nodeClass.toArray.toSeq, clue)
    assertEquals(actual.edgeClass.toArray.toSeq, expected.edgeClass.toArray.toSeq, clue)
    assertEquals(actual.links.toArray.map(_.toSeq).toSeq, expected.links.toArray.map(_.toSeq).toSeq, clue)
    Seq((actual.xs, expected.xs), (actual.ys, expected.ys)).foreach: (found, wanted) =>
      found.toArray.zip(wanted.toArray).zipWithIndex.foreach: (pair, node) =>
        pair._1.zip(pair._2).foreach((at, expected) => assertEqualsFloat(at, expected, 1e-5f, s"$clue node $node"))

  /** Everything a permuted batch holds, as the host sees it. */
  private def read(records: RecordBatch[Drawing, Node, Edge]) =
    (
      records.nodeClass.toArray.map(_.toSeq).toSeq,
      records.edgeClass.toArray.map(_.toSeq).toSeq,
      records.links.toArray.map(_.map(_.toSeq).toSeq).toSeq
    )
