import dataset.NodeClass
import dataset.NodeClasses
import dataset.NodeLink
import dataset.NodePoint
import dataset.Point
import dataset.Record
import dataset.RecordEdge
import dataset.RecordGraph
import dataset.RecordNode
import dataset.RecordScoring
import dimwit.*
import dimwit.optimizer.Adam
import munit.FunSuite

/** What an embedding may attend to, and what the loss accepts. */
class RecordSuite extends FunSuite:

  override def beforeAll(): Unit = dimwit.initialize()

  private val nodes = Axis[Node] -> 4
  private val canvas = 8

  test("the mask is exactly what remaining-node prediction needs"):
    val taken = 2
    val mask = DecoderSequence.mask(slotted(annotation(0.1f, 0.2f), annotation(0.3f, 0.4f))).toArray
    for
      target <- 0 until 8
      source <- 0 until 8
    do
      val itself = target == source
      val expected =
        if source >= 4 then itself                                             // a guess is read by no one but itself
        else if target < 4 then itself || (source <= target && source < taken)  // a node embedding carries itself and what came before it
        else source < target - 4 && source < taken                             // a prediction embedding sees only what is taken before it
      assertEquals(mask(target)(source), expected, s"row $target, column $source")

  test("no row is fully masked, since a fully masked row has no softmax"):
    for taken <- 0 to 3 do
      val mask = DecoderSequence.mask(slotted(Seq.fill(taken)(annotation(0.1f, 0.2f))*)).toArray
      mask.zipWithIndex.foreach((row, index) => assert(row.exists(identity), s"row $index attends to nothing"))

  test("the loss accepts any remaining node, and no taken one"):
    val target = slotted(annotation(0.1f, 0.2f), annotation(0.3f, 0.4f))
    val loss = RemainingNodeLoss(VType[Float32], canvas)
    def cost(answers: RecordNode*) = loss(D2G.Prediction(scored(slotted(answers*)), scored(target)), target).item

    val inOrder = cost(annotation(0.1f, 0.2f), annotation(0.3f, 0.4f), stop)
    val reversed = cost(annotation(0.3f, 0.4f), annotation(0.3f, 0.4f), stop)
    val repeated = cost(annotation(0.1f, 0.2f), annotation(0.1f, 0.2f), stop)
    val runsOn = cost(annotation(0.1f, 0.2f), annotation(0.3f, 0.4f), annotation(0.5f, 0.6f))

    assert(inOrder < 0.1f, s"a valid transcription still costs $inOrder")
    assertEqualsFloat(reversed, inOrder, 0.05f)
    assert(repeated > inOrder + 1f, s"answering with a node already taken costs $repeated, barely more than $inOrder")
    assert(runsOn > inOrder + 1f, s"running past the record costs $runsOn, barely more than $inOrder")

  private def annotation(x: Float, y: Float) = RecordNode(NodeClass.Annotation, Seq(Point(x, y)))

  private def stop = RecordNode(NodeClass.NoNode, Seq.empty)

  /** A layout that need not be a record: any node may sit at any position. */
  private def slotted(perNode: RecordNode*): Record[Node] =
    val padded = perNode ++ Seq.fill(nodes.size - perNode.length)(stop)
    def coordinate(of: Point => Float) = Tensor2(nodes.axis, Axis[NodePoint], VType[Float32]).fromArray(
      padded.map(node => Array.tabulate(NodeClass.maxPoints)(node.points.lift(_).fold(0f)(of))).toArray
    )
    Record(
      nodeClass = Tensor1(nodes.axis, VType[Int32]).fromArray(padded.map(_.nodeClass.id).toArray),
      xs = coordinate(_.x),
      ys = coordinate(_.y),
      links = Tensor2(nodes.axis, Axis[NodeLink], VType[Int32]).fromArray(
        padded.map(_ => Array.fill(NodeClass.maxLinks)(0)).toArray
      )
    )

  /** Scores that say each position holds exactly what the given layout has in it. */
  private def scored(record: Record[Node]): NodeLogits[Float32] =
    val confident = 12f
    def oneHot(value: Int, over: Int) = Array.tabulate(over)(candidate => if candidate == value then confident else 0f)
    def pixels(coordinates: Tensor2[Node, NodePoint, Float32]) =
      Tensor3(nodes.axis, Axis[NodePoint], Axis[Pixel], VType[Float32])
        .fromArray(Pixels.of(coordinates, canvas).toArray.map(_.map(oneHot(_, canvas))))
    NodeLogits(
      nodeClass = Tensor2(nodes.axis, Axis[NodeClasses], VType[Float32])
        .fromArray(record.nodeClass.toArray.map(oneHot(_, NodeClass.values.length))),
      xs = pixels(record.xs),
      ys = pixels(record.ys),
      links = Tensor3(nodes.axis, Axis[NodeLink], Axis[LinkedNode], VType[Float32])
        .fromArray(record.links.toArray.map(_.map(oneHot(_, nodes.size))))
    )

/** The whole model on a drawing far smaller than the dataset's, so that a training run of it fits
  * in a test: two lines meeting in a corner, in a 32×32 image.
  */
class D2GSuite extends FunSuite:

  override def beforeAll(): Unit = dimwit.initialize()

  override def munitTimeout: scala.concurrent.duration.Duration = scala.concurrent.duration.Duration(10, "min")

  private val canvas = 32
  private val nodes = Axis[Node] -> 6
  private val params = D2G.Params.init(numLayers = 2, numHeads = 2, embedding = 32, nodes = nodes.size, patchSize = 16, canvas = canvas, key = Random.Key(0))
  private val model = D2G(params)

  private val record = RecordGraph(
    nodes = Seq(
      RecordNode(NodeClass.Line, Seq(Point(4f / canvas, 4f / canvas), Point(4f / canvas, 26f / canvas))),
      RecordNode(NodeClass.Line, Seq(Point(4f / canvas, 26f / canvas), Point(26f / canvas, 26f / canvas)))
    ),
    edges = Seq(RecordEdge(NodeClass.Connected, 0, 1))
  )

  private val document = Tensor3(Axis[Width], Axis[Height], Axis[Channel], VType[Float32]).fromArray(
    Array.tabulate(canvas, canvas, 1): (x, y, _) =>
      val onOutline = (x == 4 && y >= 4 && y <= 26) || (y == 26 && x >= 4 && x <= 26)
      if onOutline then 0f else 1f
  )

  test("every position is scored for its class, every coordinate and every link"):
    val prediction = model(document, record.record(nodes))
    assertEquals(prediction.remaining.nodeClass.shape.dimensions.toSeq, Seq(nodes.size, NodeClass.values.length))
    assertEquals(prediction.remaining.xs.shape.dimensions.toSeq, Seq(nodes.size, NodeClass.maxPoints, canvas))
    assertEquals(prediction.remaining.links.shape.dimensions.toSeq, Seq(nodes.size, NodeClass.maxLinks, nodes.size))
    assertEquals(prediction.taken.nodeClass.shape.dimensions.toSeq, prediction.remaining.nodeClass.shape.dimensions.toSeq)

  test("training on one drawing learns to transcribe it"):
    val loss = RemainingNodeLoss(VType[Float32], canvas)
    val optimizer = Adam(learningRate = Tensor0(3e-3f))
    val step = jit: (params: D2G.Params[Float32], state: dimwit.optimizer.AdamState[D2G.Params[Float32]], target: Record[Node]) =>
      val (cost, gradients) = Autodiff.valueAndGrad((p: D2G.Params[Float32]) => loss(D2G(p)(document, target), target))(params)
      val (next, nextState) = optimizer.update(gradients, params, state)
      (cost, next, nextState)

    // A fresh linearization every step, so nothing can be learned about the order.
    val random = scala.util.Random(1)
    val (first, trained) = (1 to 800).foldLeft((Option.empty[Float], (params, optimizer.init(params)))):
      case ((first, (params, state)), _) =>
        val (cost, next, nextState) = step(params, state, record.permuted(random).record(nodes))
        (first.orElse(Some(cost.item)), (next, nextState))

    val (finalParams, _) = trained
    val linearized = record.permuted(random).record(nodes)
    val last = loss(D2G(finalParams)(document, linearized), linearized).item
    assert(last < first.get * 0.01f, s"the loss barely moved: ${first.get} -> $last")

    val transcribed = Transcriber(D2G(finalParams), nodes)(document)
    assert(RecordScoring.score(record, transcribed, tolerance = 0.5f / canvas).isExact, s"transcribed $transcribed instead of $record")
