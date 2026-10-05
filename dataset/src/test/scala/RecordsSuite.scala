package dataset

import dataset.DrawingDataset.Split
import dimwit.*
import munit.FunSuite

/** Records: laid out, permuted, read back, drawn, scored and written down without losing what they hold. */
class RecordsSuite extends FunSuite:

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
      RecordNode(NodeClass.Line, false, Seq(Point(0.2f, 0.4f), Point(0.8f, 0.4f))),
      RecordNode(NodeClass.Line, false, Seq(Point(0.8f, 0.4f), Point(0.8f, 0.9f))),
      RecordNode(NodeClass.Annotation, false, Seq(Point(0.5f, 0.3f)))
    ),
    edges = Seq(
      RecordEdge(EdgeClass.Connected, 0, 1),
      RecordEdge(EdgeClass.Annotates, 2, 0)
    )
  )

  test("a run's metrics are appended one row per checkpoint and tolerance, blank where nothing was measured"):
    val detected = RecordScoring.Scored(nodes = 10, nodesFound = 9, nodesPredicted = 12, nodesExact = false, relationships = 0, relationshipsFound = 0, relationshipsPredicted = 0, isExact = false)
    val rows = Seq(
      Metrics.Row(10000, None, 8f, Seq(detected, detected.copy(nodesExact = true), detected.copy(nodesExact = true, isExact = true))),
      Metrics.Row(20000, Some(0.5f), 8f, Seq(detected))
    )
    val csv = Metrics.Csv("test", Corpus.LShape, "xs", parameters = 123, trainingSeconds = Some(45))
    rows.foreach(row => csv.append(Seq(row)))
    val lines = java.nio.file.Files.readAllLines(csv.path)
    java.nio.file.Files.delete(csv.path)
    assertEquals(lines.get(0), "model,corpus,size,step,threshold,tolerance,parameters,training_seconds,node_recall,node_precision,nodes_exact,edge_recall,edge_precision,records_exact")
    assertEquals(lines.get(1), "test,l-shape,xs,10000,,8,123,45,90.00,75.00,66.67,,,33.33")
    assertEquals(lines.get(2), "test,l-shape,xs,20000,0.5,8,123,45,90.00,75.00,0.00,,,0.00")

  test("a transcript keeps the predicted nodes in their order and says which target each stands for"):
    val pixel = 1f / Corpus.SketchGraph.canvas
    val target = RecordGraph(
      Seq(
        RecordNode(NodeClass.Line, false, Seq(Point(10 * pixel, 20 * pixel), Point(50 * pixel, 20 * pixel))),
        RecordNode(NodeClass.Circle, false, Seq(Point(100 * pixel, 100 * pixel), Point(140 * pixel, 100 * pixel)))
      ),
      Seq.empty
    )
    val threePixelsOff = RecordNode(NodeClass.Line, false, Seq(Point(13 * pixel, 20 * pixel), Point(50 * pixel, 20 * pixel)))
    val written = RecordGraph(Seq(target.nodes(1), threePixelsOff, RecordNode(NodeClass.Circle, false, Seq(Point(0f, 0f), Point(0.1f, 0f)))), Seq.empty)
    val path = Transcripts.write("test", Corpus.LShape, "xs", 1000, Seq((target, written)))
    val line = java.nio.file.Files.readAllLines(path).get(0)
    java.nio.file.Files.delete(path)
    assert(line.contains(""""points": [[10.00, 20.00], [50.00, 20.00]]"""), line)
    assert(line.contains(""""matched": {"0": [1, null, null], "2": [1, null, null], "4": [1, 0, null], "8": [1, 0, null]}"""), line)

  test("an arc keeps its middle through being laid out, permuted on the device and read back"):
    val bent = RecordGraph(
      Seq(
        RecordNode(NodeClass.Line, false, Seq(Point(0.2f, 0.5f), Point(0.6f, 0.5f))),
        RecordNode(NodeClass.Arc, false, Seq(Point(0.2f, 0.5f), Point(0.6f, 0.5f), Point(0.4f, 0.3f)))
      ),
      Seq.empty
    )
    assertEquals(RecordGraph.of(bent.record(nodes, edges)), bent)
    val permuted = RecordBatch.of(Seq(bent), Axis[Drawing], nodes, edges).permuted(dimwit.Random.Key(3), nodes, edges)
    assertEquals(RecordGraph.of(permuted).head.nodes.toSet, bent.nodes.toSet)

  test("construction geometry stays construction geometry, and is only found as such"):
    val dashed = RecordNode(NodeClass.Line, true, Seq(Point(0.2f, 0.5f), Point(0.6f, 0.5f)))
    val sketch = RecordGraph(Seq(dashed, RecordNode(NodeClass.Point, false, Seq(Point(0.4f, 0.5f)))), Seq.empty)
    val permuted = RecordBatch.of(Seq(sketch), Axis[Drawing], nodes, Axis[Edge] -> 0).permuted(dimwit.Random.Key(5), nodes, Axis[Edge] -> 0)
    assertEquals(RecordGraph.of(permuted).head.nodes.toSet, sketch.nodes.toSet)
    val drawnSolid = sketch.copy(nodes = Seq(dashed.copy(isConstruction = false), sketch.nodes(1)))
    assertEquals(RecordScoring.score(sketch, drawnSolid, tolerance = 0f).nodesFound, 1)

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
        assertEquals(classes.sortBy(held => if held.isNode then 0 else 1), classes, s"seed $seed, drawing $at: nodes before empty positions")
      permuted.edgeClass.toArray.zipWithIndex.foreach: (drawing, at) =>
        val classes = drawing.map(EdgeClass.fromId).toSeq
        assertEquals(classes.sortBy(held => if held.isEdge then 0 else 1), classes, s"seed $seed, drawing $at: relationships before empty positions")
      permuted.edgeClass.toArray.lazyZip(permuted.subject.toArray).lazyZip(permuted.obj.toArray).foreach: (classes, subjects, objs) =>
        classes.lazyZip(subjects).lazyZip(objs).foreach: (held, subject, obj) =>
          val edgeClass = EdgeClass.fromId(held)
          if edgeClass.isSymmetric then assert(subject < obj, s"$edgeClass relates $subject to $obj rather than its ends in ascending order")
          if !edgeClass.isEdge then assertEquals((subject, obj), (0, 0), "an empty position relates nothing")

  test("a record is drawn as the picture it stands for"):
    val canvas = 32
    val blank = greyLevels(Tensor3(Axis[Width] -> canvas, Axis[Height] -> canvas, Axis[Channel] -> 1, VType[Float32]).fill(1f))
    val drawn = RecordDrawing(record, blank, Axis[Channel]).asInt(VType[Int32]).toArray
    assertEquals(drawn.length, canvas)
    assertEquals(drawn.head.head.toSeq, Seq(255, 255, 255), "a corner the record does not reach stays blank")
    // The record's first line runs from (0.2, 0.4) to (0.8, 0.4), so the middle of the canvas is on it.
    assertEquals(drawn(canvas / 2)(math.round(0.4f * canvas)).toSeq, Seq(20, 60, 190), "the middle of a line is drawn in the line's colour")

  test("a record is drawn where the drawing it was read from has its ink"):
    val data = DrawingDataset.open(Corpus.LShape)(Axis[Width], Axis[Height], Axis[Channel], Axis[Node], Axis[Edge])(Split.Validation)
    data.samples.take(3).zipWithIndex.foreach: (sample, index) =>
      val drawing = greyLevels(sample.image)
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

  /** What a record's relationships say in terms of its nodes rather than of their slots. A
    * symmetric relationship says nothing by which of its two nodes comes first.
    */
  private def related(record: RecordGraph): Set[(EdgeClass, Set[RecordNode], Seq[RecordNode])] =
    record.edges.map: edge =>
      val ends = Seq(record.nodes(edge.subject), record.nodes(edge.obj))
      if edge.edgeClass.isSymmetric then (edge.edgeClass, ends.toSet, Seq.empty) else (edge.edgeClass, Set.empty[RecordNode], ends)
    .toSet

  /** Everything a permuted batch holds, as the host sees it. */
  private def read(records: RecordBatch[Drawing, Node, Edge]) =
    (
      records.nodeClass.toArray.map(_.toSeq).toSeq,
      records.edgeClass.toArray.map(_.toSeq).toSeq,
      records.subject.toArray.map(_.toSeq).toSeq,
      records.obj.toArray.map(_.toSeq).toSeq
    )
