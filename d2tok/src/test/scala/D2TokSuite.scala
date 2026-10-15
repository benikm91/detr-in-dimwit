import d2tok.*
import d2tok.eval.Transcriber
import d2tok.model.*
import dataset.NodeClass
import dataset.Point
import dataset.RecordGraph
import dataset.RecordNode
import dimwit.*
import dimwit.stats.Uniform

class D2TokSuite extends munit.FunSuite:

  override def beforeAll(): Unit = dimwit.initialize()

  private val canvas = 64
  private val words = Words(canvas)
  private def at(pixel: Int) = pixel.toFloat / canvas
  private val sketch = RecordGraph(
    Seq(
      RecordNode(NodeClass.Line, false, Seq(Point(at(3), at(40)), Point(at(60), at(40)))),
      RecordNode(NodeClass.Point, true, Seq(Point(at(10), at(11)))),
      RecordNode(NodeClass.Arc, false, Seq(Point(at(12), at(30)), Point(at(36), at(30)), Point(at(24), at(18)))),
      RecordNode(NodeClass.Circle, true, Seq(Point(at(20), at(50)), Point(at(30), at(50))))
    ),
    Seq.empty
  )
  private val slots = 5
  private lazy val tokens = Axis[Token] -> words.tokens(slots)
  private def drawing(seed: Int) =
    val pixels = Shape3(Axis[Width] -> canvas, Axis[Height] -> canvas, Axis[Channel] -> 1)
    Uniform(Tensor(pixels).fill(0f), Tensor(pixels).fill(1f)).sample(Random.Key(seed))

  test("a record written in words reads back as the same nodes, followed by its end"):
    val (written, count) = words.written(sketch.record(Axis[Node] -> slots, Axis[Relationship] -> 0).nodes, tokens)
    val said = written.toArray.toSeq
    assertEquals(count.item, 5 + 3 + 7 + 5 + 1, "a line, a point, an arc and a circle, then the end")
    assertEquals(said.drop(count.item - 1).distinct, Seq(words.end), "the end, and nothing but the end after it")
    assertEquals(words.read(said), sketch.nodes)

  test("words are read up to the first one that does not fit where it stands"):
    val said = words.written(sketch.record(Axis[Node] -> slots, Axis[Relationship] -> 0).nodes, tokens)._1.toArray.toSeq
    assertEquals(words.read(said.updated(5, words.end)), sketch.nodes.take(1), "a point cut short by the end")
    assertEquals(words.read(said.updated(1, said.head)), Seq.empty, "a type where a coordinate belongs")

  test("writing one word at a time with caches answers as the whole sequence read at once"):
    val model = D2Tok(D2Tok.Params.init(numLayers = 2, numHeads = 2, embedding = 32, words = words.size, tokens = tokens.size, key = Random.Key(0)))
    val document = drawing(1)
    val written = words.written(sketch.record(Axis[Node] -> slots, Axis[Relationship] -> 0).nodes, tokens)._1
    val read = concatenate(Tensor0(words.start).prependAxis(Axis[Token]), written.slice(Axis[Token].at(0 until tokens.size - 1)), Axis[Token])
    val whole = model.logits(document, read)
    val documentCache = model.read(model.encodeDocument(document))
    val said = read.toArray
    (0 until tokens.size).foldLeft(model.nothingWritten(tokens)): (cached, position) =>
      val (scores, nowWritten) = model.answerAt(documentCache, cached, position, said(position))
      val difference = (scores - whole.slice(Axis[Token].at(position))).abs.max.item
      assert(difference < 1e-4f, s"position $position: the cached scores are off by $difference")
      nowWritten

  test("the transcriber writes what reading the whole sequence again at every position would"):
    val params = D2Tok.Params.init(numLayers = 2, numHeads = 2, embedding = 32, words = words.size, tokens = tokens.size, key = Random.Key(2))
    val model = D2Tok(params)
    val documents = (3 to 5).map(drawing)
    val readWhole = jit((document: Tensor3[Width, Height, Channel, Float32], read: Tensor1[Token, Int32]) => model.logits(document, read))

    /** The words of a transcription, every one read from the whole sequence before it again. */
    def readAgainAtEveryPosition(document: Tensor3[Width, Height, Channel, Float32]): Seq[Int] =
      (0 until tokens.size).foldLeft(Seq.empty[Int]): (said, position) =>
        val read = Tensor1(tokens.axis, VType[Int32]).fromArray((words.start +: said).padTo(tokens.size, words.end).take(tokens.size).toArray)
        said :+ readWhole(document, read).slice(Axis[Token].at(position)).argmax(Axis[Word]).item

    val written = Transcriber(words, tokens, drawings = documents.size).written(params, documents)
    documents.lazyZip(written).zipWithIndex.foreach:
      case ((document, said), at) =>
        assertEquals(said, readAgainAtEveryPosition(document), s"drawing $at")
