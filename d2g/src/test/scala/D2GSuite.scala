package d2g

import d2g.model.*
import d2g.train.*
import d2g.eval.*
import d2g.config.*
import dataset.EdgeClass
import dataset.EdgeClasses
import dataset.NodeClass
import dataset.NodeClasses
import dataset.Point
import dataset.Record
import dataset.RecordEdge
import dataset.RecordEdges
import dataset.RecordGraph
import dataset.RecordNode
import dataset.RecordNodes
import dataset.RecordScoring
import EdgeHead.EdgeLogits
import NodeHead.NodeLogits
import dimwit.*
import dimwit.optimizer.Adam
import munit.FunSuite
import deepwit.init.Init

/** What an embedding may attend to, and what the loss accepts. */
/** How many queries a slot is asked with in these tests. */
private val askedAtOnce = 2

class RecordSuite extends FunSuite:

  override def beforeAll(): Unit = dimwit.initialize()

  private val nodes = Axis[Node] -> 4
  private val edges = Axis[Edge] -> 4
  private val canvas = 8

  test("the mask is exactly what remaining-node prediction needs"):
    val slots = 4
    val mask = jointSequenceMask(askedAtOnce)(joined(slots)).toArray
    for
      target <- 0 until joined(slots).size
      source <- 0 until joined(slots).size
    do
      val expected =
        // a guess is read by no one but itself, its own token's twin included
        if source >= slots then target == source
        // a taken embedding carries itself and what came before it
        else if target < slots then source <= target
        // a prediction embedding sees only what is taken before the slot it answers for, which is
        // where it sits within its own token's block
        else source < (target - slots) % slots
      assertEquals(mask(target)(source), expected, s"row $target, column $source")

  test("no row is fully masked, since a fully masked row has no softmax"):
    for slots <- 1 to 4 do
      val mask = jointSequenceMask(askedAtOnce)(joined(slots)).toArray
      mask.zipWithIndex.foreach((row, index) => assert(row.exists(identity), s"row $index of $slots attends to nothing"))

  test("the node loss accepts any remaining node, and no taken one"):
    val target = slotted(annotation(0.1f, 0.2f), annotation(0.3f, 0.4f))
    val loss = RemainingNodeLoss(VType[Float32], canvas)
    // The two prediction tokens of a slot answer separately, so a cost takes what each said.
    def cost(one: Seq[RecordNode], other: Seq[RecordNode]) =
      loss(D2G.NodeQueryLogits.of(Seq(scored(slotted(one*)), scored(slotted(other*)))), target).item

    val first = Seq(annotation(0.1f, 0.2f), annotation(0.3f, 0.4f), noNode)
    val second = Seq(annotation(0.3f, 0.4f), annotation(0.3f, 0.4f), noNode)
    val apart = cost(first, second)
    val together = cost(first, first)
    val repeated = cost(Seq(annotation(0.1f, 0.2f), annotation(0.1f, 0.2f), noNode), second)
    val runsOn = cost(Seq(annotation(0.1f, 0.2f), annotation(0.3f, 0.4f), annotation(0.5f, 0.6f)), second)

    assert(apart < 0.1f, s"two different remaining nodes still cost $apart")
    assert(together > apart + 0.5f, s"both tokens naming the same node costs $together, barely more than $apart")
    assert(repeated > apart + 1f, s"answering with a node already taken costs $repeated, barely more than $apart")
    assert(runsOn > apart + 1f, s"running past the nodes costs $runsOn, barely more than $apart")

  test("the edge loss accepts any remaining relationship, and no taken one"):
    val target = related(RecordEdge(EdgeClass.Connected, 0, 1), RecordEdge(EdgeClass.Annotates, 2, 0))
    val loss = RemainingEdgeLoss(VType[Float32])
    def cost(one: Seq[RecordEdge], other: Seq[RecordEdge]) =
      loss(D2G.EdgeQueryLogits.of(Seq(scored(related(one*)), scored(related(other*)))), target).item

    val first = Seq(RecordEdge(EdgeClass.Connected, 0, 1), RecordEdge(EdgeClass.Annotates, 2, 0), noEdge)
    val second = Seq(RecordEdge(EdgeClass.Annotates, 2, 0), RecordEdge(EdgeClass.Annotates, 2, 0), noEdge)
    val inOrder = cost(first, second)
    val reversed = cost(second, first)
    val repeated = cost(Seq(RecordEdge(EdgeClass.Connected, 0, 1), RecordEdge(EdgeClass.Connected, 0, 1), noEdge), second)
    val runsOn = cost(Seq(RecordEdge(EdgeClass.Connected, 0, 1), RecordEdge(EdgeClass.Annotates, 2, 0), RecordEdge(EdgeClass.Annotates, 1, 2)), second)

    assert(inOrder < 0.1f, s"a valid transcription still costs $inOrder")
    assertEqualsFloat(reversed, inOrder, 0.05f)
    assert(repeated > inOrder + 1f, s"answering with a relationship already taken costs $repeated, barely more than $inOrder")
    assert(runsOn > inOrder + 1f, s"running past the relationships costs $runsOn, barely more than $inOrder")

  test("a relationship reads the nodes that are there, and not the positions past them"):
    val (patches, embedding, mixed) = (Axis[Patch] -> 5, Axis[Embedding] -> 8, Axis[EmbeddingMixed] -> 16)
    val (held, padded) = (Axis[Node] -> 2, Axis[Node] -> 4)
    val decoder = EdgeDecoder(EdgeDecoder.Params.xavierUniformDepthScaled(numBlocks = 1, numHeads = 2, embedding, embedding, mixed, Random.Key(3)))
    val document = Init.xavierUniform(patches, embedding, Random.Key(4))
    val taken = Init.xavierUniform(edges, embedding, Random.Key(5))
    val asked = stack(Seq.tabulate(askedAtOnce)(query => Init.xavierUniform(edges, embedding, Random.Key(6 + query))), Axis[PoolQuery])
    val nodes = Init.xavierUniform(padded, embedding, Random.Key(7))

    def answered(nodes: Tensor2[Node, Embedding, Float32], holdsNode: Seq[Boolean]) =
      val mask = Tensor1(nodes.shape.extent(Axis[Node]), VType[Bool]).fromArray(holdsNode.toArray)
      decoder.forTraining(document, NodeSource(nodes, mask), taken, asked)._2.flatten.toArray.toSeq

    val withPadding = answered(nodes, Seq(true, true, false, false))
    val withoutPadding = answered(nodes.slice(Axis[Node].at(0 until held.size)), Seq(true, true))
    withPadding.zip(withoutPadding).zipWithIndex.foreach: (both, at) =>
      assertEqualsFloat(both._1, both._2, 1e-5f, s"answer $at")

  /** The joined sequence of `slots` taken embeddings and as many prediction embeddings, which is
    * the extent the mask is asked for.
    */
  private def joined(slots: Int) = Axis[NodeDecoder.Context] -> (1 + askedAtOnce) * slots

  private def annotation(x: Float, y: Float) = RecordNode(NodeClass.Annotation, Seq(Point(x, y)))

  private def noNode = RecordNode(NodeClass.NoNode, Seq.empty)

  private def noEdge = RecordEdge(EdgeClass.NoEdge, 0, 0)

  /** A layout that need not be a record: any node may sit at any position. */
  private def slotted(perNode: RecordNode*): RecordNodes[Node] =
    val padded = perNode ++ Seq.fill(nodes.size - perNode.length)(noNode)
    def placed(of: Point => Float, at: Int) = Tensor1(nodes.axis, VType[Float32]).fromArray(
      padded.map(_.points.lift(at).fold(0f)(of)).toArray
    )
    RecordNodes(
      nodeClass = Tensor1(nodes.axis, VType[Int32]).fromArray(padded.map(_.nodeClass.id).toArray),
      startX = placed(_.x, 0),
      startY = placed(_.y, 0),
      endX = placed(_.x, 1),
      endY = placed(_.y, 1)
    )

  /** The same for relationships: any relationship may sit at any position. */
  private def related(perEdge: RecordEdge*): RecordEdges[Edge] =
    val padded = perEdge ++ Seq.fill(edges.size - perEdge.length)(noEdge)
    def named(end: RecordEdge => Int) = Tensor1(edges.axis, VType[Int32]).fromArray(padded.map(end).toArray)
    RecordEdges(
      edgeClass = Tensor1(edges.axis, VType[Int32]).fromArray(padded.map(_.edgeClass.id).toArray),
      subject = named(_.subject),
      obj = named(_.obj)
    )

  /** Scores that say each position holds exactly what the given layout has in it. */
  private def scored(taken: RecordNodes[Node]): NodeLogits[Float32] =
    def pixels(coordinate: Tensor1[Node, Float32]) =
      Tensor2(nodes.axis, Axis[Pixel], VType[Float32])
        .fromArray(Pixels.of(coordinate, canvas).toArray.map(oneHot(_, canvas)))
    NodeLogits(
      nodeClass = Tensor2(nodes.axis, Axis[NodeClasses], VType[Float32])
        .fromArray(taken.nodeClass.toArray.map(oneHot(_, NodeClass.values.length))),
      startX = pixels(taken.startX),
      startY = pixels(taken.startY),
      endX = pixels(taken.endX),
      endY = pixels(taken.endY)
    )

  private def scored(taken: RecordEdges[Edge]): EdgeLogits[Float32] =
    def named(end: Tensor1[Edge, Int32]) =
      Tensor2(edges.axis, Axis[LinkedNode], VType[Float32]).fromArray(end.toArray.map(oneHot(_, nodes.size)))
    EdgeLogits(
      edgeClass = Tensor2(edges.axis, Axis[EdgeClasses], VType[Float32])
        .fromArray(taken.edgeClass.toArray.map(oneHot(_, EdgeClass.values.length))),
      subject = named(taken.subject),
      obj = named(taken.obj)
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
  private val params = D2G.Params.init(numLayers = 2, numHeads = 2, embedding = 32, nodes = nodes.size, edges = edges.size, queries = 2, canvas = canvas, key = Random.Key(0))
  private val model = D2G(params)
  private val asked = Random.Key(7)

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
    val scored = model.logits(document, record.record(nodes, edges), asked)
    Seq(scored.nodes.at(0), scored.nodes.at(1)).foreach: guess =>
      assertEquals(guess.nodeClass.shape.dimensions.toSeq, Seq(nodes.size, NodeClass.values.length))
      assertEquals(guess.startX.shape.dimensions.toSeq, Seq(nodes.size, canvas))
    Seq(scored.edges.at(0), scored.edges.at(1)).foreach: guess =>
      assertEquals(guess.edgeClass.shape.dimensions.toSeq, Seq(edges.size, EdgeClass.values.length))
      assertEquals(guess.subject.shape.dimensions.toSeq, Seq(edges.size, nodes.size))

  test("the whole pool answers at once, along the query axis"):
    val scored = model.logitsPerQuery(model.encodeDocument(document), record.record(nodes, edges))
    assertEquals(scored.nodes.nodeClass.shape.dimensions.toSeq, Seq(askedAtOnce, nodes.size, NodeClass.values.length))
    assertEquals(scored.edges.edgeClass.shape.dimensions.toSeq, Seq(askedAtOnce, edges.size, EdgeClass.values.length))

  test("the training state carries a new linearization on to every step"):
    val optimizer = deepwit.optimizer.LearningRateScheduler(
      lr => dimwit.optimizer.AdamW(Adam(learningRate = lr), Tensor0(0f)),
      deepwit.optimizer.ConstantLearningRate(Tensor0(1e-3f))
    )
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
      val scored = D2G(params).logits(document, target, asked)
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
