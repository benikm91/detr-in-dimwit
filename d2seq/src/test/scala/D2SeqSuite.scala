import d2seq.*
import d2seq.eval.Transcriber
import d2seq.model.*
import d2seq.model.NodeHead.NodeLogits
import d2seq.train.NextNodeLoss
import dataset.NodeClass
import dataset.NodeClasses
import dataset.RecordGraph
import dataset.RecordNode
import dataset.RecordNodes
import dimwit.*
import dimwit.stats.Uniform

class D2SeqSuite extends munit.FunSuite:

  override def beforeAll(): Unit = dimwit.initialize()

  private val canvas = 64
  private val pixels = Shape3(Axis[Width] -> canvas, Axis[Height] -> canvas, Axis[Channel] -> 1)
  private def drawing(seed: Int) = Uniform(Tensor(pixels).fill(0f), Tensor(pixels).fill(1f)).sample(Random.Key(seed))
  private def field(values: Float*) = Tensor1(Axis[Node], VType[Float32]).fromArray(values.toArray)
  private lazy val record = RecordNodes(
    nodeClass = Tensor1(Axis[Node], VType[Int32]).fromArray(Array(NodeClass.Line, NodeClass.Arc, NodeClass.Circle, NodeClass.Point, NodeClass.NoNode).map(_.id)),
    construction = Tensor1(Axis[Node], VType[Int32]).fromArray(Array(0, 1, 0, 0, 0)),
    startX = field(0.1f, 0.2f, 0.3f, 0.4f, 0f),
    startY = field(0.5f, 0.6f, 0.7f, 0.8f, 0f),
    endX = field(0.9f, 0.1f, 0.2f, 0f, 0f),
    endY = field(0.4f, 0.5f, 0.6f, 0f, 0f),
    midX = field(0f, 0.8f, 0f, 0f, 0f),
    midY = field(0f, 0.9f, 0f, 0f, 0f)
  )
  private lazy val slots = record.nodeClass.shape(Axis[Node])
  private def one[W](values: Tensor1[Node, W], slot: Int) = values.slice(Axis[Node].at(slot, 1))
  private def nodeAt(nodes: RecordNodes[Node], slot: Int) = RecordNodes(one(nodes.nodeClass, slot), one(nodes.construction, slot), one(nodes.startX, slot),
    one(nodes.startY, slot), one(nodes.endX, slot), one(nodes.endY, slot), one(nodes.midX, slot), one(nodes.midY, slot))

  test("writing one node at a time with caches answers as the whole sequence read at once"):
    val model = D2Seq(D2Seq.Params.init(numLayers = 2, numHeads = 2, embedding = 32, nodes = slots, canvas = canvas, key = Random.Key(0)))
    val document = drawing(1)
    val whole = model.logits(document, record)
    val documentCache = model.read(model.encodeDocument(document))
    (0 until slots).foldLeft((model.nothingWritten(Axis[Node] -> slots), model.start)):
      case ((written, previous), slot) =>
        val (atSlot, nowWritten) = model.answerAt(documentCache, written, slot, previous)
        def assertSame[L: Label](name: String, cached: Tensor2[Node, L, Float32], read: Tensor2[Node, L, Float32]) =
          val difference = (cached.squeeze(Axis[Node]) - read.slice(Axis[Node].at(slot))).abs.max.item
          assert(difference < 1e-4f, s"slot $slot, $name: the cached answer is off by $difference")
        assertSame("class", atSlot.nodeClass, whole.nodeClass)
        assertSame("construction", atSlot.construction, whole.construction)
        assertSame("start x", atSlot.startX, whole.startX)
        assertSame("end y", atSlot.endY, whole.endY)
        assertSame("middle y", atSlot.midY, whole.midY)
        (nowWritten, model.embed(nodeAt(record, slot)))

  test("the transcriber writes what reading the whole sequence again at every slot would"):
    val untrained = D2Seq.Params.init(numLayers = 2, numHeads = 2, embedding = 32, nodes = slots, canvas = canvas, key = Random.Key(2))
    // An untrained model stops at once; one that never says it is done writes every slot.
    val classHead = untrained.head.nodeClass
    val params = untrained.copy(head = untrained.head.copy(nodeClass = classHead.copy(bias = classHead.bias.set(Axis[NodeClasses].at(NodeClass.NoNode.id))(-100f))))
    val model = D2Seq(params)
    val documents = (3 to 5).map(drawing)
    val readWhole = jit((document: Tensor3[Width, Height, Channel, Float32], written: RecordNodes[Node]) => model.logits(document, written))

    /** The previous transcription: at every slot, the whole sequence so far read again. */
    def readAgainAtEverySlot(document: Tensor3[Width, Height, Channel, Float32]): Seq[RecordNode] =
      def laidOut(written: Seq[RecordNode]) = RecordGraph(written, Seq.empty).record(Axis[Node] -> slots, Axis[Relationship] -> 0).nodes
      (0 until slots).foldLeft((Seq.empty[RecordNode], true)):
        case ((written, writing), _) if !writing => (written, false)
        case ((written, _), slot) =>
          val said = model.nodeHead.decide(readWhole(document, laidOut(written)))
          val at = Axis[Node].at(slot)
          val nodeClass = NodeClass.fromId(said.nodeClass.slice(at).item)
          def point(x: Tensor1[Node, Float32], y: Tensor1[Node, Float32]) = dataset.Point(x.slice(at).item, y.slice(at).item)
          val next = RecordNode(nodeClass, said.construction.slice(at).item == 1,
            Seq(point(said.startX, said.startY), point(said.endX, said.endY), point(said.midX, said.midY)).take(nodeClass.numPoints))
          if nodeClass.isNode then (written :+ next, true) else (written, false)
      ._1

    val transcribed = Transcriber(Axis[Node] -> slots, drawings = documents.size)(params, documents)
    assert(transcribed.forall(_.nodes.size == slots), s"wrote ${transcribed.map(_.nodes.size)} nodes, not one in every slot")
    documents.lazyZip(transcribed).zipWithIndex.foreach:
      case ((document, written), at) =>
        val expected = readAgainAtEverySlot(document)
        assertEquals(written.nodes.size, expected.size, s"drawing $at")
        written.nodes.lazyZip(expected).foreach: (cached, again) =>
          assertEquals(cached.nodeClass, again.nodeClass, s"drawing $at")
          assertEquals(cached.isConstruction, again.isConstruction, s"drawing $at")
          cached.points.lazyZip(again.points).foreach((a, b) => assert(math.abs(a.x - b.x) < 1e-6 && math.abs(a.y - b.y) < 1e-6, s"drawing $at: $cached against $again"))

  test("the loss charges every slot up to the one that says the record ends, and none after it"):
    val longer = RecordNodes(
      nodeClass = Tensor1(Axis[Node], VType[Int32]).fromArray(Array(NodeClass.Line, NodeClass.Point, NodeClass.NoNode, NodeClass.NoNode).map(_.id)),
      construction = Tensor1(Axis[Node], VType[Int32]).fromArray(Array(0, 0, 0, 0)),
      startX = field(0.1f, 0.4f, 0f, 0f),
      startY = field(0.5f, 0.8f, 0f, 0f),
      endX = field(0.9f, 0f, 0f, 0f),
      endY = field(0.4f, 0f, 0f, 0f),
      midX = field(0f, 0f, 0f, 0f),
      midY = field(0f, 0f, 0f, 0f)
    )
    val model = D2Seq(D2Seq.Params.init(numLayers = 1, numHeads = 2, embedding = 32, nodes = 4, canvas = canvas, key = Random.Key(6)))
    val answered = model.logits(drawing(7), longer)
    val loss = NextNodeLoss(VType[Float32], canvas)
    def changedAt(slot: Int, logits: NodeLogits[Float32]) = logits.copy(nodeClass = logits.nodeClass.set(Axis[Node].at(slot))(logits.nodeClass.slice(Axis[Node].at(slot)) *! 3f))
    val asGiven = loss(answered, longer).item
    assertEqualsFloat(loss(changedAt(3, answered), longer).item, asGiven, 1e-6f, "the slot after the end slot is charged")
    assert(math.abs(loss(changedAt(2, answered), longer).item - asGiven) > 1e-3f, "the end slot is not charged")
