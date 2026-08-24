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

  private val nodes = Axis[Node] -> 8
  private val geometry = Geometry(minimumSize = 4f / 256, annotationSize = 12f / 256)

  private val record = RecordGraph(
    nodes = Seq(
      RecordNode(NodeClass.Line, Seq(Point(0.2f, 0.4f), Point(0.8f, 0.4f))),
      RecordNode(NodeClass.Line, Seq(Point(0.8f, 0.4f), Point(0.8f, 0.9f))),
      RecordNode(NodeClass.Annotation, Seq(Point(0.5f, 0.3f)))
    ),
    edges = Seq(
      RecordEdge(NodeClass.Connected, 0, 1),
      RecordEdge(NodeClass.Annotates, 2, 0)
    )
  )

  test("a line is drawn as the box between its end points, an annotation as a box around it"):
    val objects = Objects.of(record.record(nodes), geometry)
    val box = objects.detection.box
    assertEquals(objects.detection.label.toArray.toSeq, Seq(1, 1, 2, 0, 0, 0, 0, 0))
    assertEqualsFloat(box.centerX.toArray(0), 0.5f, 1e-6f)
    assertEqualsFloat(box.width.toArray(0), 0.6f, 1e-6f)
    assertEqualsFloat(box.height.toArray(0), geometry.minimumSize, 1e-6f)
    assertEqualsFloat(box.width.toArray(2), geometry.annotationSize, 1e-6f)
    assert(box.width.toArray.drop(3).forall(_ == 0f), "a relationship is not drawn")

  test("a symmetric relationship is drawn both ways round, a directed one is not"):
    val relations = Objects.of(record.record(nodes), geometry).relations.toArray
    assertEquals(relations(0)(1)(RelationClass.Connected.id), 1f)
    assertEquals(relations(1)(0)(RelationClass.Connected.id), 1f)
    assertEquals(relations(2)(0)(RelationClass.Annotates.id), 1f)
    assertEquals(relations(0)(2)(RelationClass.Annotates.id), 0f)
    assertEquals(relations.flatten.flatten.sum, 3f)

  test("a record survives being permuted, laid out and read back"):
    val random = scala.util.Random(7)
    for _ <- 1 to 20 do
      val permuted = RecordGraph.of(record.permuted(random).record(nodes))
      assertEquals(permuted.nodes.toSet, record.nodes.toSet)
      assertEquals(related(permuted), related(record))

  test("the objects of a record hold the record"):
    assertSameRecord(Objects.record(Objects.of(record.record(nodes), geometry)), record.record(nodes))

  Split.values.foreach: split =>
    test(s"the objects of every record of the ${split.fileName} split hold that record"):
      val data = LShapeDataset.open(Axis[Width], Axis[Height], Axis[Channel], Axis[Node])(split)
      data.samples().take(200).zipWithIndex.foreach: (sample, index) =>
        assertSameRecord(Objects.record(Objects.of(sample.target, data.geometry)), sample.target, s"sample $index")

  /** What a record's relationships say in terms of its nodes rather than of their slots. A
    * symmetric relationship says nothing by which of its two nodes comes first.
    */
  private def related(record: RecordGraph): Set[(NodeClass, Set[RecordNode], Seq[RecordNode])] =
    record.edges.map: edge =>
      val ends = Seq(record.nodes(edge.subject), record.nodes(edge.obj))
      if edge.edgeClass.isSymmetric then (edge.edgeClass, ends.toSet, Seq.empty) else (edge.edgeClass, Set.empty[RecordNode], ends)
    .toSet

  private def assertSameRecord(actual: Record[Node], expected: Record[Node], clue: String = ""): Unit =
    assertEquals(actual.nodeClass.toArray.toSeq, expected.nodeClass.toArray.toSeq, clue)
    assertEquals(actual.links.toArray.map(_.toSeq).toSeq, expected.links.toArray.map(_.toSeq).toSeq, clue)
    Seq((actual.xs, expected.xs), (actual.ys, expected.ys)).foreach: (found, wanted) =>
      found.toArray.zip(wanted.toArray).zipWithIndex.foreach: (pair, node) =>
        pair._1.zip(pair._2).foreach((at, expected) => assertEqualsFloat(at, expected, 1e-5f, s"$clue node $node"))
