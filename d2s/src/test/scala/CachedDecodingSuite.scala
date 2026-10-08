import d2s.*
import d2s.eval.Transcriber
import d2s.eval.answered
import d2s.model.*
import d2s.model.NodeHead.NodeLogits
import dataset.Corpus
import dataset.DrawingDataset
import dataset.DrawingDataset.Split
import dataset.NodeClass
import dataset.RecordGraph
import dataset.RecordNode
import dataset.RecordNodes
import dimwit.*
import dimwit.stats.Uniform

class CachedDecodingSuite extends munit.FunSuite:

  override def beforeAll(): Unit = dimwit.initialize()

  test("decoding one slot at a time with caches answers as the whole record read at once"):
    val canvas = 64
    val slots = 5
    val params = D2S.Params.init(numLayers = 2, numHeads = 2, embedding = 32, nodes = slots, queries = 3, canvas = canvas, key = Random.Key(0))
    val model = D2S(params)
    val pixels = Shape3(Axis[Width] -> canvas, Axis[Height] -> canvas, Axis[Channel] -> 1)
    val drawing = Uniform(Tensor(pixels).fill(0f), Tensor(pixels).fill(1f)).sample(Random.Key(1))
    def field(values: Float*) = Tensor1(Axis[Node], VType[Float32]).fromArray(values.toArray)
    val record = RecordNodes(
      nodeClass = Tensor1(Axis[Node], VType[Int32]).fromArray(Array(NodeClass.Line, NodeClass.Arc, NodeClass.Circle, NodeClass.Line, NodeClass.NoNode).map(_.id)),
      construction = Tensor1(Axis[Node], VType[Int32]).fromArray(Array(0, 1, 0, 0, 0)),
      startX = field(0.1f, 0.2f, 0.3f, 0.4f, 0f),
      startY = field(0.5f, 0.6f, 0.7f, 0.8f, 0f),
      endX = field(0.9f, 0.1f, 0.2f, 0.3f, 0f),
      endY = field(0.4f, 0.5f, 0.6f, 0.7f, 0f),
      midX = field(0f, 0.8f, 0f, 0f, 0f),
      midY = field(0f, 0.9f, 0f, 0f, 0f)
    )
    val encoded = model.encodeDocument(drawing)
    val whole = model.logitsPerQuery(encoded, record)
    val documentCache = model.read(encoded)

    def one[V](values: Tensor1[Node, V], slot: Int) = values.slice(Axis[Node].at(slot, 1))
    (0 until slots).foldLeft(model.nothingTaken(Axis[Node] -> slots)): (taken, slot) =>
      val atSlot = model.answerAt(documentCache, taken, slot)
      def assertSame[L: Label](name: String, cached: Tensor2[Node, L, Float32], read: Tensor3[PoolQuery, Node, L, Float32]) =
        val difference = (cached.relabel(Axis[Node] -> Axis[PoolQuery]) - read.slice(Axis[Node].at(slot))).abs.max.item
        assert(difference < 1e-4f, s"slot $slot, $name: the cached answer is off by $difference")
      assertSame("class", atSlot.nodeClass, whole.nodeClass)
      assertSame("construction", atSlot.construction, whole.construction)
      assertSame("start x", atSlot.startX, whole.startX)
      assertSame("end y", atSlot.endY, whole.endY)
      assertSame("middle y", atSlot.midY, whole.midY)
      val node = RecordNodes(one(record.nodeClass, slot), one(record.construction, slot), one(record.startX, slot), one(record.startY, slot),
        one(record.endX, slot), one(record.endY, slot), one(record.midX, slot), one(record.midY, slot))
      val (_, nowTaken) = model.take(documentCache, taken, slot, node)
      nowTaken

  test("the transcriber writes what reading the whole record at every slot would"):
    val corpus = Corpus.VitruvionPrimitives
    val nodes = Axis[Node] -> (corpus.maxNodes + 1)
    val untrained = D2S.Params.init(numLayers = 2, numHeads = 2, embedding = 32, nodes = nodes.size, queries = 3, canvas = corpus.canvas, key = Random.Key(2))
    // An untrained model stops at once; one that never says it is done writes every slot.
    val classHead = untrained.nodes.head.nodeClass
    val neverDone = classHead.copy(bias = classHead.bias.set(Axis[dataset.NodeClasses].at(NodeClass.NoNode.id))(-100f))
    val params = untrained.copy(nodes = untrained.nodes.copy(head = untrained.nodes.head.copy(nodeClass = neverDone)))
    val model = D2S(params)
    val data = DrawingDataset.open(corpus)(Axis[Width], Axis[Height], Axis[Channel], Axis[Node], Axis[Relationship])(Split.Validation)
    val documents = data.samples.take(4).map(_.image).toSeq
    val read = jit((document: Tensor3[Width, Height, Channel, Float32], taken: RecordNodes[Node]) => model.logitsPerQuery(model.encodeDocument(document), taken))

    /** The previous transcription: at every slot, the whole record so far read again. */
    def readAgainAtEverySlot(document: Tensor3[Width, Height, Channel, Float32]): Seq[RecordNode] =
      def laidOut(written: Seq[RecordNode]) = RecordGraph(written, Seq.empty).record(nodes, Axis[Relationship] -> 1).nodes
      (0 until nodes.size).foldLeft((Seq.empty[RecordNode], true)):
        case ((written, writing), slot) if !writing => (written, false)
        case ((written, _), slot) =>
          val scored = read(document, laidOut(written))
          def candidates[L: Label](logits: Tensor3[PoolQuery, Node, L, Float32]) = logits.slice(Axis[Node].at(slot)).relabel(Axis[PoolQuery] -> Axis[Node])
          val (said, score) = answered(model.nodeHead, NodeLogits(candidates(scored.nodeClass), candidates(scored.construction), candidates(scored.startX),
            candidates(scored.startY), candidates(scored.endX), candidates(scored.endY), candidates(scored.midX), candidates(scored.midY)))
          val at = Axis[Node].at(score.argmax(Axis[Node]).item)
          val nodeClass = NodeClass.fromId(said.nodeClass.slice(at).item)
          def point(x: Tensor1[Node, Float32], y: Tensor1[Node, Float32]) = dataset.Point(x.slice(at).item, y.slice(at).item)
          val likeliest = RecordNode(nodeClass, said.construction.slice(at).item == 1,
            Seq(point(said.startX, said.startY), point(said.endX, said.endY), point(said.midX, said.midY)).take(nodeClass.numPoints))
          if likeliest.nodeClass.isNode then (written :+ likeliest, true) else (written, false)
      ._1

    val transcribed = Transcriber(nodes, drawings = documents.size)(params, documents)
    assert(transcribed.forall(_.nodes.size == nodes.size), s"wrote ${transcribed.map(_.nodes.size)} nodes, not one in every slot")
    documents.lazyZip(transcribed).zipWithIndex.foreach:
      case ((document, written), at) =>
        val expected = readAgainAtEverySlot(document)
        assertEquals(written.nodes.size, expected.size, s"drawing $at")
        written.nodes.lazyZip(expected).foreach: (cached, again) =>
          assertEquals(cached.nodeClass, again.nodeClass, s"drawing $at")
          assertEquals(cached.isConstruction, again.isConstruction, s"drawing $at")
          cached.points.lazyZip(again.points).foreach((a, b) => assert(math.abs(a.x - b.x) < 1e-6 && math.abs(a.y - b.y) < 1e-6, s"drawing $at: $cached against $again"))
