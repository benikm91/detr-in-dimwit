import d2s.*
import d2s.model.*
import d2s.eval.Transcriber
import dataset.Canvas
import dimwit.*

class TranscriberSuite extends munit.FunSuite:
  test("a batch of drawings is transcribed into as many records"):
    dimwit.initialize()
    val params = D2S.Params.init(numLayers = 1, numHeads = 2, embedding = 32, nodes = 5, queries = 2, canvas = Canvas, key = Random.Key(0))
    val blank = Tensor(Shape3(Axis[Width] -> 256, Axis[Height] -> 256, Axis[Channel] -> 1), VType[Float32]).fill(0f)
    val records = Transcriber(Axis[Node] -> 5, drawings = 4)(params, Seq(blank, blank, blank))
    assertEquals(records.size, 3)
    assert(records.forall(_.edges.isEmpty))
