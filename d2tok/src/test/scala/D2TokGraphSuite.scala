import d2tok.*
import d2tok.floorplan.SequenceOrder
import d2tok.floorplangraph.*
import d2tok.floorplangraph.config.D2TokGraphSetup
import d2tok.floorplangraph.eval.Transcriber
import d2tok.floorplangraph.model.D2TokGraph
import dataset.Point
import dimwit.*
import dimwit.stats.Uniform
import syntheticFloorplansDataset.FloorplanClass
import syntheticFloorplansDataset.FloorplanEdge
import syntheticFloorplansDataset.FloorplanGraph
import syntheticFloorplansDataset.FloorplanNode
import syntheticFloorplansDataset.FloorplanRecordBatch
import syntheticFloorplansDataset.FloorplanRelationship.In
import syntheticFloorplansDataset.FloorplanRelationship.Joined

class D2TokGraphSuite extends munit.FunSuite:

  override def beforeAll(): Unit = dimwit.initialize()

  private trait Drawing derives Label

  private val canvas = 64
  private def at(x: Int, y: Int) = Point(x.toFloat / canvas, y.toFloat / canvas)
  private val plan = FloorplanGraph(
    Seq(
      FloorplanNode(FloorplanClass.Window, at(4, 20), at(4, 30)),
      FloorplanNode(FloorplanClass.OutsideWall, at(4, 4), at(60, 4)),
      FloorplanNode(FloorplanClass.InsideWall, at(30, 4), at(30, 60)),
      FloorplanNode(FloorplanClass.OutsideWall, at(4, 4), at(4, 60)),
      FloorplanNode(FloorplanClass.Door, at(30, 10), at(30, 18))
    ),
    Seq(FloorplanEdge(Joined, 1, 3), FloorplanEdge(In, 4, 2), FloorplanEdge(Joined, 1, 2), FloorplanEdge(In, 0, 3))
  )
  private val (nodes, relationships) = (Axis[Node] -> 6, Axis[Relationship] -> 5)
  private val setup = D2TokGraphSetup(numLayers = 2, numHeads = 2, embedding = 32, canvas = canvas, checkpointRoot = "")
  private val words = RelationshipWords(nodes.size)
  private lazy val tokens = Axis[RelationshipToken] -> words.tokens(relationships.size)

  private def records = FloorplanRecordBatch.of(Seq(plan), Axis[Drawing], nodes, relationships)

  /** Every relationship by the nodes it links, which does not change with the order they are
    * written in; a symmetric one links its two either way round.
    */
  private def linked(graph: FloorplanGraph) = graph.edges.map: edge =>
    val pair = Seq(graph.nodes(edge.subject), graph.nodes(edge.obj))
    (edge.relationship, if edge.relationship.isSymmetric then pair.toSet else pair)
  .toSet

  test("relationships written in words read back as the same relationships, followed by their end"):
    val (written, count) = words.written(plan.record(nodes, relationships).edges, tokens)
    val said = written.toArray.toSeq
    assertEquals(count.item, 4 * 3 + 1, "four relationships of three words, then the end")
    assertEquals(said.drop(count.item - 1).distinct, Seq(words.end), "the end, and nothing but the end after it")
    assertEquals(words.read(said), plan.edges)
    assertEquals(words.read(said.updated(4, words.end)), plan.edges.take(1), "a relationship cut short by the end")

  test("by nodes, the relationships are written by the number of their first node, then of their second"):
    val written = FloorplanGraph.of(RelationshipOrder.ByNodes.laidOut(records, Random.Key(0))).head
    assertEquals(written.edges, Seq(FloorplanEdge(In, 0, 3), FloorplanEdge(Joined, 1, 2), FloorplanEdge(Joined, 1, 3), FloorplanEdge(In, 4, 2)))

  test("however the nodes and relationships are laid out, the relationships link the same nodes"):
    val orders = for
      nodeOrder <- Seq(SequenceOrder.Random, SequenceOrder.WallsFirst)
      relationshipOrder <- RelationshipOrder.values.toSeq
      seed <- 1 to 3
    yield
      val (forNodes, forRelationships) = Random.Key(seed).splitToTuple(2)
      val written = FloorplanGraph.of(relationshipOrder.laidOut(nodeOrder.laidOut(records, forNodes, canvas), forRelationships)).head
      assertEquals(linked(written), linked(plan), s"${nodeOrder.name}, ${relationshipOrder.name}, seed $seed")
      written.edges
    assert(orders.distinct.size > 1, "the random orders differ")

  private def drawing(seed: Int) =
    val pixels = Shape3(Axis[Width] -> canvas, Axis[Height] -> canvas, Axis[Channel] -> 1)
    Uniform(Tensor(pixels).fill(0f), Tensor(pixels).fill(1f)).sample(Random.Key(seed))

  private def initialParams(seed: Int) = D2TokGraph.Params.init(
    numLayers = 2, numHeads = 2, embedding = 32,
    nodeWords = setup.words.size, nodeTokens = setup.tokens, nodes = syntheticFloorplansDataset.SyntheticFloorplans.maxNodes,
    relationshipWords = setup.relationshipWords.size, relationshipTokens = setup.relationshipTokens,
    key = Random.Key(seed)
  )

  test("the transcriber writes the relationships that reading all of them again at every position would"):
    val params = initialParams(1)
    val model = D2TokGraph(params)
    val documents = (2 to 3).map(drawing)
    val transcribed = Transcriber(setup, drawings = documents.size).written(params, documents)
    val nodeTokens = Axis[Token] -> setup.tokens
    val relationshipTokens = Axis[RelationshipToken] -> setup.relationshipTokens
    val readWhole = jit: (document: Tensor3[Width, Height, Channel, Float32], readNodes: Tensor1[Token, Int32], present: Tensor1[Node, Bool], readRelationships: Tensor1[RelationshipToken, Int32]) =>
      model.logits(document, readNodes, setup.words.perNode, present, readRelationships)._2

    documents.lazyZip(transcribed).zipWithIndex.foreach:
      case ((document, (nodeWords, relationshipWords)), at) =>
        val readNodes = Tensor1(nodeTokens.axis, VType[Int32]).fromArray((setup.words.start +: nodeWords).take(nodeTokens.size).toArray)
        val firstEnd = nodeWords.indexOf(setup.words.end) match
          case -1  => nodeTokens.size
          case end => end
        val present = Tensor1(Axis[Node], VType[Bool]).fromArray(Array.tabulate(syntheticFloorplansDataset.SyntheticFloorplans.maxNodes)(node => (node + 1) * setup.words.perNode <= firstEnd))
        val readAgain = (0 until relationshipTokens.size).foldLeft(Seq.empty[Int]): (said, position) =>
          val read = Tensor1(relationshipTokens.axis, VType[Int32]).fromArray((setup.relationshipWords.start +: said).padTo(relationshipTokens.size, setup.relationshipWords.end).take(relationshipTokens.size).toArray)
          said :+ readWhole(document, readNodes, present, read).slice(Axis[RelationshipToken].at(position)).argmax(Axis[Word]).item
        assertEquals(relationshipWords, readAgain, s"drawing $at")
