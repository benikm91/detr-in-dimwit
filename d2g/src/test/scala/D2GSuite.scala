import dataset.EdgeClass
import dataset.EdgeClasses
import dataset.NodeClass
import dataset.NodeClasses
import dataset.NodeLink
import dataset.NodePoint
import dataset.Point
import dataset.Record
import dataset.RecordEdge
import dataset.RecordEdges
import dataset.RecordGraph
import dataset.RecordNode
import dataset.RecordNodes
import dataset.RecordScoring
import dimwit.*
import dimwit.optimizer.Adam
import munit.FunSuite

/** What an embedding may attend to, and what the loss accepts. */
class RecordSuite extends FunSuite:

  override def beforeAll(): Unit = dimwit.initialize()

  private val nodes = Axis[Node] -> 4
  private val edges = Axis[Edge] -> 4
  private val canvas = 8

  test("the mask is exactly what remaining-node prediction needs"):
    val mask = jointSequenceMask(joined(slots = 4)).toArray
    for
      target <- 0 until 8
      source <- 0 until 8
    do
      val itself = target == source
      val expected =
        if source >= 4 then itself                          // a guess is read by no one but itself
        else if target < 4 then itself || source <= target   // a taken embedding carries itself and what came before it
        else source < target - 4                             // a prediction embedding sees only what is taken before it
      assertEquals(mask(target)(source), expected, s"row $target, column $source")

  test("no row is fully masked, since a fully masked row has no softmax"):
    for slots <- 1 to 4 do
      val mask = jointSequenceMask(joined(slots)).toArray
      mask.zipWithIndex.foreach((row, index) => assert(row.exists(identity), s"row $index of $slots attends to nothing"))

  test("the node loss accepts any remaining node, and no taken one"):
    val target = slotted(annotation(0.1f, 0.2f), annotation(0.3f, 0.4f))
    val loss = RemainingNodeLoss(VType[Float32], canvas)
    def cost(answers: RecordNode*) = loss(D2G.NodeScores(remaining = scored(slotted(answers*)), taken = scored(target)), target).item

    val inOrder = cost(annotation(0.1f, 0.2f), annotation(0.3f, 0.4f), noNode)
    val reversed = cost(annotation(0.3f, 0.4f), annotation(0.3f, 0.4f), noNode)
    val repeated = cost(annotation(0.1f, 0.2f), annotation(0.1f, 0.2f), noNode)
    val runsOn = cost(annotation(0.1f, 0.2f), annotation(0.3f, 0.4f), annotation(0.5f, 0.6f))

    assert(inOrder < 0.1f, s"a valid transcription still costs $inOrder")
    assertEqualsFloat(reversed, inOrder, 0.05f)
    assert(repeated > inOrder + 1f, s"answering with a node already taken costs $repeated, barely more than $inOrder")
    assert(runsOn > inOrder + 1f, s"running past the nodes costs $runsOn, barely more than $inOrder")

  test("the edge loss accepts any remaining relationship, and no taken one"):
    val target = related(RecordEdge(EdgeClass.Connected, 0, 1), RecordEdge(EdgeClass.Annotates, 2, 0))
    val loss = RemainingEdgeLoss(VType[Float32])
    def cost(answers: RecordEdge*) = loss(D2G.EdgeScores(remaining = scored(related(answers*)), taken = scored(target)), target).item

    val inOrder = cost(RecordEdge(EdgeClass.Connected, 0, 1), RecordEdge(EdgeClass.Annotates, 2, 0), noEdge)
    val reversed = cost(RecordEdge(EdgeClass.Annotates, 2, 0), RecordEdge(EdgeClass.Annotates, 2, 0), noEdge)
    val repeated = cost(RecordEdge(EdgeClass.Connected, 0, 1), RecordEdge(EdgeClass.Connected, 0, 1), noEdge)
    val runsOn = cost(RecordEdge(EdgeClass.Connected, 0, 1), RecordEdge(EdgeClass.Annotates, 2, 0), RecordEdge(EdgeClass.Annotates, 1, 2))

    assert(inOrder < 0.1f, s"a valid transcription still costs $inOrder")
    assertEqualsFloat(reversed, inOrder, 0.05f)
    assert(repeated > inOrder + 1f, s"answering with a relationship already taken costs $repeated, barely more than $inOrder")
    assert(runsOn > inOrder + 1f, s"running past the relationships costs $runsOn, barely more than $inOrder")

  test("a relationship reads the nodes that are there, and not the positions past them"):
    val (patches, embedding, mixed) = (Axis[Patch] -> 5, Axis[Embedding] -> 8, Axis[EmbeddingMixed] -> 16)
    val (held, padded) = (Axis[Node] -> 2, Axis[Node] -> 4)
    val decoder = EdgeDecoder(EdgeDecoder.Params.xavierUniformDepthScaled(numBlocks = 1, numHeads = 2, embedding, embedding, mixed, Random.Key(3)))
    val document = deepwit.init.Init.xavierUniform(patches, embedding, Random.Key(4))
    val taken = deepwit.init.Init.xavierUniform(edges, embedding, Random.Key(5))
    val predictions = deepwit.init.Init.xavierUniform(edges, embedding, Random.Key(6)).relabel(Axis[Edge] -> Axis[EdgeDecoder.Prediction])
    val nodes = deepwit.init.Init.xavierUniform(padded, embedding, Random.Key(7))

    def answered(nodes: Tensor2[Node, Embedding, Float32], holdsNode: Seq[Boolean]) =
      val mask = Tensor1(nodes.shape.extent(Axis[Node]), VType[Bool]).fromArray(holdsNode.toArray)
      decoder.forTraining(document, nodes, mask, taken, predictions)._2.toArray.map(_.toSeq).toSeq

    val withPadding = answered(nodes, Seq(true, true, false, false))
    val withoutPadding = answered(nodes.slice(Axis[Node].at(0 until held.size)), Seq(true, true))
    withPadding.zip(withoutPadding).zipWithIndex.foreach: (both, slot) =>
      both._1.zip(both._2).foreach((found, wanted) => assertEqualsFloat(found, wanted, 1e-5f, s"slot $slot"))

  /** The joined sequence of `slots` taken embeddings and as many prediction embeddings, which is
    * the extent the mask is asked for.
    */
  private def joined(slots: Int) = Axis[NodeDecoder.Context] -> 2 * slots

  private def annotation(x: Float, y: Float) = RecordNode(NodeClass.Annotation, Seq(Point(x, y)))

  private def noNode = RecordNode(NodeClass.NoNode, Seq.empty)

  private def noEdge = RecordEdge(EdgeClass.NoEdge, 0, 0)

  /** A layout that need not be a record: any node may sit at any position. */
  private def slotted(perNode: RecordNode*): RecordNodes[Node] =
    val padded = perNode ++ Seq.fill(nodes.size - perNode.length)(noNode)
    def coordinate(of: Point => Float) = Tensor2(nodes.axis, Axis[NodePoint], VType[Float32]).fromArray(
      padded.map(node => Array.tabulate(NodeClass.maxPoints)(node.points.lift(_).fold(0f)(of))).toArray
    )
    RecordNodes(
      nodeClass = Tensor1(nodes.axis, VType[Int32]).fromArray(padded.map(_.nodeClass.id).toArray),
      xs = coordinate(_.x),
      ys = coordinate(_.y)
    )

  /** The same for relationships: any relationship may sit at any position. */
  private def related(perEdge: RecordEdge*): RecordEdges[Edge] =
    val padded = perEdge ++ Seq.fill(edges.size - perEdge.length)(noEdge)
    RecordEdges(
      edgeClass = Tensor1(edges.axis, VType[Int32]).fromArray(padded.map(_.edgeClass.id).toArray),
      links = Tensor2(edges.axis, Axis[NodeLink], VType[Int32]).fromArray(padded.map(edge => Array(edge.subject, edge.obj)).toArray)
    )

  /** Scores that say each position holds exactly what the given layout has in it. */
  private def scored(taken: RecordNodes[Node]): NodeLogits[Float32] =
    def pixels(coordinates: Tensor2[Node, NodePoint, Float32]) =
      Tensor3(nodes.axis, Axis[NodePoint], Axis[Pixel], VType[Float32])
        .fromArray(Pixels.of(coordinates, canvas).toArray.map(_.map(oneHot(_, canvas))))
    NodeLogits(
      nodeClass = Tensor2(nodes.axis, Axis[NodeClasses], VType[Float32])
        .fromArray(taken.nodeClass.toArray.map(oneHot(_, NodeClass.values.length))),
      xs = pixels(taken.xs),
      ys = pixels(taken.ys)
    )

  private def scored(taken: RecordEdges[Edge]): EdgeLogits[Float32] =
    EdgeLogits(
      edgeClass = Tensor2(edges.axis, Axis[EdgeClasses], VType[Float32])
        .fromArray(taken.edgeClass.toArray.map(oneHot(_, EdgeClass.values.length))),
      links = Tensor3(edges.axis, Axis[NodeLink], Axis[LinkedNode], VType[Float32])
        .fromArray(taken.links.toArray.map(_.map(oneHot(_, nodes.size))))
    )

  private def oneHot(value: Int, over: Int) =
    val confident = 12f
    Array.tabulate(over)(candidate => if candidate == value then confident else 0f)

/** The whole model on a drawing far smaller than the dataset's, so that a training run of it fits
  * in a test: two lines meeting in a corner, in a 32×32 image.
  */
class D2GSuite extends FunSuite:

  override def beforeAll(): Unit = dimwit.initialize()

  override def munitTimeout: scala.concurrent.duration.Duration = scala.concurrent.duration.Duration(10, "min")

  private val canvas = 32
  private val nodes = Axis[Node] -> 4
  private val edges = Axis[Edge] -> 3
  private val params = D2G.Params.init(numLayers = 2, numHeads = 2, embedding = 32, nodes = nodes.size, edges = edges.size, patchSize = 16, canvas = canvas, key = Random.Key(0))
  private val model = D2G(params)

  private val record = RecordGraph(
    nodes = Seq(
      RecordNode(NodeClass.Line, Seq(Point(4f / canvas, 4f / canvas), Point(4f / canvas, 26f / canvas))),
      RecordNode(NodeClass.Line, Seq(Point(4f / canvas, 26f / canvas), Point(26f / canvas, 26f / canvas)))
    ),
    edges = Seq(RecordEdge(EdgeClass.Connected, 0, 1))
  )

  private val document = Tensor3(Axis[Width], Axis[Height], Axis[Channel], VType[Float32]).fromArray(
    Array.tabulate(canvas, canvas, 1): (x, y, _) =>
      val onOutline = (x == 4 && y >= 4 && y <= 26) || (y == 26 && x >= 4 && x <= 26)
      if onOutline then 0f else 1f
  )

  test("every position is scored for what its half of a record carries"):
    val scored = model(document, record.record(nodes, edges))
    assertEquals(scored.nodes.remaining.nodeClass.shape.dimensions.toSeq, Seq(nodes.size, NodeClass.values.length))
    assertEquals(scored.nodes.remaining.xs.shape.dimensions.toSeq, Seq(nodes.size, NodeClass.maxPoints, canvas))
    assertEquals(scored.nodes.taken.nodeClass.shape.dimensions.toSeq, scored.nodes.remaining.nodeClass.shape.dimensions.toSeq)
    assertEquals(scored.edges.remaining.edgeClass.shape.dimensions.toSeq, Seq(edges.size, EdgeClass.values.length))
    assertEquals(scored.edges.remaining.links.shape.dimensions.toSeq, Seq(edges.size, EdgeClass.maxLinks, nodes.size))
    assertEquals(scored.edges.taken.edgeClass.shape.dimensions.toSeq, scored.edges.remaining.edgeClass.shape.dimensions.toSeq)

  test("the training state carries a new linearization on to every step"):
    val optimizer = Adam(learningRate = Tensor0(1e-3f))
    val advance = jit: (state: D2GTrainState) =>
      val (next, _) = state.linearization.split2()
      state.copy(linearization = next)
    val start = D2GTrainState(params, optimizer.init(params), Random.Key(42), Tensor0(-1f))
    val once = advance(start)
    val twice = advance(once)
    assertNotEquals(once.linearization, start.linearization, "the key a compiled step hands on is the one it was given")
    assertNotEquals(twice.linearization, once.linearization, "every step draws from where the last one left off")

  test("training on one drawing learns to transcribe it"):
    val nodeLoss = RemainingNodeLoss(VType[Float32], canvas)
    val edgeLoss = RemainingEdgeLoss(VType[Float32])
    def cost(params: D2G.Params[Float32], target: Record[Node, Edge]) =
      val scored = D2G(params)(document, target)
      nodeLoss(scored.nodes, target.nodes) + edgeLoss(scored.edges, target.edges)

    val optimizer = Adam(learningRate = Tensor0(3e-3f))
    val step = jit: (params: D2G.Params[Float32], state: dimwit.optimizer.AdamState[D2G.Params[Float32]], target: Record[Node, Edge]) =>
      val (lastCost, gradients) = Autodiff.valueAndGrad((p: D2G.Params[Float32]) => cost(p, target))(params)
      val (next, nextState) = optimizer.update(gradients, params, state)
      (lastCost, next, nextState)

    // A fresh linearization every step, so nothing can be learned about the order.
    val random = scala.util.Random(1)
    val (first, trained) = (1 to 800).foldLeft((Option.empty[Float], (params, optimizer.init(params)))):
      case ((first, (params, state)), _) =>
        val (lastCost, next, nextState) = step(params, state, record.permuted(random).record(nodes, edges))
        (first.orElse(Some(lastCost.item)), (next, nextState))

    val (finalParams, _) = trained
    val linearized = record.permuted(random).record(nodes, edges)
    val last = cost(finalParams, linearized).item
    assert(last < first.get * 0.01f, s"the loss barely moved: ${first.get} -> $last")

    val transcribed = Transcriber(D2G(finalParams), nodes, edges)(document)
    assert(RecordScoring.score(record, transcribed, tolerance = 0.5f / canvas).isExact, s"transcribed $transcribed instead of $record")
