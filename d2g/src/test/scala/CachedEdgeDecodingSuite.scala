import d2g.*
import d2g.eval.Transcriber
import d2g.model.*
import d2g.model.EdgeHead.EdgeLogits
import d2s.eval.chosen
import d2s.model.PoolQuery
import dataset.EdgeClass
import dataset.EdgeClasses
import dataset.NodeClass
import dataset.NodeClasses
import dataset.Record
import dataset.RecordEdges
import dataset.RecordGraph
import dataset.RecordNodes
import dimwit.*
import dimwit.stats.Uniform

class CachedEdgeDecodingSuite extends munit.FunSuite:

  override def beforeAll(): Unit = dimwit.initialize()

  test("decoding one relationship at a time with caches answers as the whole record read at once"):
    val canvas = 64
    val (nodeSlots, edgeSlots) = (5, 4)
    val params = D2G.Params.init(numLayers = 2, numHeads = 2, embedding = 32, nodes = nodeSlots, edges = edgeSlots, queries = 3, canvas = canvas, key = Random.Key(0))
    val model = D2G(params)
    val pixels = Shape3(Axis[Width] -> canvas, Axis[Height] -> canvas, Axis[Channel] -> 1)
    val drawing = Uniform(Tensor(pixels).fill(0f), Tensor(pixels).fill(1f)).sample(Random.Key(1))
    def field(values: Float*) = Tensor1(Axis[Node], VType[Float32]).fromArray(values.toArray)
    val nodes = RecordNodes(
      nodeClass = Tensor1(Axis[Node], VType[Int32]).fromArray(Array(NodeClass.Line, NodeClass.Arc, NodeClass.Circle, NodeClass.Line, NodeClass.NoNode).map(_.id)),
      construction = Tensor1(Axis[Node], VType[Int32]).fromArray(Array(0, 1, 0, 0, 0)),
      startX = field(0.1f, 0.2f, 0.3f, 0.4f, 0f),
      startY = field(0.5f, 0.6f, 0.7f, 0.8f, 0f),
      endX = field(0.9f, 0.1f, 0.2f, 0.3f, 0f),
      endY = field(0.4f, 0.5f, 0.6f, 0.7f, 0f),
      midX = field(0f, 0.8f, 0f, 0f, 0f),
      midY = field(0f, 0.9f, 0f, 0f, 0f)
    )
    def ends(values: Int*) = Tensor1(Axis[Edge], VType[Int32]).fromArray(values.toArray)
    val edges = RecordEdges(
      edgeClass = Tensor1(Axis[Edge], VType[Int32]).fromArray(Array(EdgeClass.Connected, EdgeClass.Annotates, EdgeClass.Connected, EdgeClass.NoEdge).map(_.id)),
      subject = ends(0, 3, 1, 0),
      obj = ends(1, 2, 2, 0)
    )
    val encoded = model.encodeDocument(drawing)
    val whole = model.logitsPerQuery(encoded, Record(nodes, edges)).edges

    def one[V](values: Tensor1[Node, V], slot: Int) = values.slice(Axis[Node].at(slot, 1))
    val documentCache = model.set.read(encoded)
    val (carried, _) = (0 until nodeSlots).foldLeft((Seq.empty[Tensor1[Embedding, Float32]], model.set.nothingTaken(Axis[Node] -> nodeSlots))):
      case ((carried, taken), slot) =>
        val node = RecordNodes(one(nodes.nodeClass, slot), one(nodes.construction, slot), one(nodes.startX, slot), one(nodes.startY, slot),
          one(nodes.endX, slot), one(nodes.endY, slot), one(nodes.midX, slot), one(nodes.midY, slot))
        val (carriedNode, nowTaken) = model.set.take(documentCache, taken, slot, node)
        (carried :+ carriedNode, nowTaken)
    val sources = model.readForEdges(encoded, NodeSource(stack(carried, Axis[Node]), !(nodes.nodeClass elementEquals_! NodeClass.NoNode.id)))

    def oneEdge(values: Tensor1[Edge, Int32], slot: Int) = values.slice(Axis[Edge].at(slot, 1))
    (0 until edgeSlots).foldLeft(model.noEdgeTaken(Axis[Edge] -> edgeSlots)): (taken, slot) =>
      val atSlot = model.answerEdgeAt(sources, taken, slot)
      def assertSame[L: Label](name: String, cached: Tensor2[Edge, L, Float32], read: Tensor3[PoolQuery, Edge, L, Float32]) =
        val difference = (cached.relabel(Axis[Edge] -> Axis[PoolQuery]) - read.slice(Axis[Edge].at(slot))).abs.max.item
        assert(difference < 1e-4f, s"slot $slot, $name: the cached answer is off by $difference")
      assertSame("class", atSlot.edgeClass, whole.edgeClass)
      assertSame("subject", atSlot.subject, whole.subject)
      assertSame("object", atSlot.obj, whole.obj)
      model.takeEdge(sources, taken, slot, RecordEdges(oneEdge(edges.edgeClass, slot), oneEdge(edges.subject, slot), oneEdge(edges.obj, slot)))

  test("the transcriber writes the nodes d2s would, and the relationships reading the whole record at every slot would"):
    val canvas = 64
    val (nodes, edges) = (Axis[Node] -> 5, Axis[Edge] -> 4)
    val untrained = D2G.Params.init(numLayers = 2, numHeads = 2, embedding = 32, nodes = nodes.size, edges = edges.size, queries = 3, canvas = canvas, key = Random.Key(2))
    // An untrained model stops at once; one that never says it is done writes every slot.
    val nodeClassHead = untrained.set.nodes.head.nodeClass
    val edgeClassHead = untrained.edges.head.edgeClass
    val params = untrained.copy(
      set = untrained.set.copy(nodes = untrained.set.nodes.copy(head = untrained.set.nodes.head.copy(nodeClass =
        nodeClassHead.copy(bias = nodeClassHead.bias.set(Axis[NodeClasses].at(NodeClass.NoNode.id))(-100f))))),
      edges = untrained.edges.copy(head = untrained.edges.head.copy(edgeClass =
        edgeClassHead.copy(bias = edgeClassHead.bias.set(Axis[EdgeClasses].at(EdgeClass.NoEdge.id))(-100f))))
    )
    val model = D2G(params)
    val pixels = Shape3(Axis[Width] -> canvas, Axis[Height] -> canvas, Axis[Channel] -> 1)
    val documents = (3 to 5).map(seed => Uniform(Tensor(pixels).fill(0f), Tensor(pixels).fill(1f)).sample(Random.Key(seed)))
    val readWhole = jit((document: Tensor3[Width, Height, Channel, Float32], taken: Record[Node, Edge]) => model.logitsPerQuery(model.encodeDocument(document), taken).edges)

    /** The relationships of `written`, read again in whole at every slot, and kept as the model wrote them. */
    def readAgainAtEverySlot(document: Tensor3[Width, Height, Channel, Float32], written: RecordNodes[Node]): RecordEdges[Edge] =
      val nothing = RecordEdges(Tensor1(edges.axis, VType[Int32]).fromArray(Array.fill(edges.size)(EdgeClass.NoEdge.id)), Tensor1(edges.axis, VType[Int32]).fromArray(Array.fill(edges.size)(0)), Tensor1(edges.axis, VType[Int32]).fromArray(Array.fill(edges.size)(0)))
      (0 until edges.size).foldLeft((nothing, true)):
        case ((taken, writing), slot) if !writing => (taken, false)
        case ((taken, _), slot) =>
          val scored = readWhole(document, Record(written, taken))
          def candidates[L: Label](logits: Tensor3[PoolQuery, Edge, L, Float32]) = logits.slice(Axis[Edge].at(slot)).relabel(Axis[PoolQuery] -> Axis[Edge])
          val logits = EdgeLogits(candidates(scored.edgeClass), candidates(scored.subject), candidates(scored.obj))
          val said = model.edgeHead.decide(logits)
          val likeliest = Axis[Edge].at((chosen(logits.edgeClass) + chosen(logits.subject) + chosen(logits.obj)).argmax(Axis[Edge]).item)
          def put(values: Tensor1[Edge, Int32], from: Tensor1[Edge, Int32]) = values.set(Axis[Edge].at(slot))(from.slice(likeliest))
          if said.edgeClass.slice(likeliest).item == EdgeClass.NoEdge.id then (taken, false)
          else (RecordEdges(put(taken.edgeClass, said.edgeClass), put(taken.subject, said.subject), put(taken.obj, said.obj)), true)
      ._1

    val transcribed = Transcriber(nodes, edges, drawings = documents.size)(params, documents)
    val setTranscribed = d2s.eval.Transcriber(nodes, drawings = documents.size)(params.set, documents)
    documents.lazyZip(transcribed).lazyZip(setTranscribed).zipWithIndex.foreach:
      case ((document, written, writtenBySet), at) =>
        assertEquals(written.nodes, writtenBySet.nodes, s"drawing $at")
        assertEquals(written.edges.size, edges.size, s"drawing $at: not a relationship in every slot")
        val writtenNodes = RecordGraph(written.nodes, Seq.empty).record(nodes, edges).nodes
        assertEquals(written.edges, RecordGraph.of(Record(writtenNodes, readAgainAtEverySlot(document, writtenNodes))).edges, s"drawing $at")
