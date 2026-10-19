package detrbucket.sketch.train

import detr.*
import detrbucket.sketch.Buckets
import detrbucket.sketch.model.BucketHead.NodeLogits
import detrbucket.sketch.model.Presence
import dataset.IsConstruction
import dataset.NodeClass
import dataset.RecordNodes
import dimwit.*
import munit.FunSuite

class BucketLossSuite extends FunSuite:

  override def beforeAll(): Unit = dimwit.initialize()

  private val canvas = 8
  /** Two queries of every class: points 0–1, lines 2–3, circles 4–5, arcs 6–7. */
  private val buckets = Buckets(Seq(NodeClass.Point, NodeClass.Line, NodeClass.Circle, NodeClass.Arc).map(_ -> 2))
  private val loss = BucketLoss(VType[Float32], canvas, buckets)

  /** A record of one line, from (2, 4) to (6, 4) in pixels, and one slot holding no node. */
  private lazy val oneLine = RecordNodes(
    nodeClass = Tensor1(Axis[Node], VType[Int32]).fromArray(Array(NodeClass.Line.id, NodeClass.NoNode.id)),
    construction = Tensor1(Axis[Node], VType[Int32]).fromArray(Array(0, 0)),
    startX = Tensor1(Axis[Node], VType[Float32]).fromArray(Array(0.25f, 0f)),
    startY = Tensor1(Axis[Node], VType[Float32]).fromArray(Array(0.5f, 0f)),
    endX = Tensor1(Axis[Node], VType[Float32]).fromArray(Array(0.75f, 0f)),
    endY = Tensor1(Axis[Node], VType[Float32]).fromArray(Array(0.5f, 0f)),
    midX = Tensor1(Axis[Node], VType[Float32]).fromArray(Array(0f, 0f)),
    midY = Tensor1(Axis[Node], VType[Float32]).fromArray(Array(0f, 0f))
  )

  /** Every query sure it answers with nothing, except those `answering`, which are sure of that line. */
  private def sureOfTheLine(answering: Int*): NodeLogits[Float32] =
    def sure[L: Label](axis: Axis[L], size: Int)(answer: Int => Int) =
      Tensor2(Axis[Query], axis, VType[Float32]).fromArray(Array.tabulate(buckets.queries, size)((query, value) => if value == answer(query) then 20f else 0f))
    def pixel(at: Int) = sure(Axis[Pixel], canvas)(query => if answering.contains(query) then at else 0)
    NodeLogits(
      presence = sure(Axis[Presence], 2)(query => if answering.contains(query) then 1 else 0),
      construction = sure(Axis[IsConstruction], 2)(_ => 0),
      startX = pixel(2),
      startY = pixel(4),
      endX = pixel(6),
      endY = pixel(4),
      midX = pixel(0),
      midY = pixel(0)
    )

  test("costs next to nothing when a line query answers the line and every other query answers none"):
    assert(loss(sureOfTheLine(3), oneLine).item < 1e-3f)

  test("does not care which of the line queries answers"):
    assertEqualsFloat(loss(sureOfTheLine(2), oneLine).item, loss(sureOfTheLine(3), oneLine).item, 1e-6f)

  test("charges when only a point query answers the line"):
    assert(loss(sureOfTheLine(0), oneLine).item > 1f)

  test("has a gradient inside jit"):
    val presence = sureOfTheLine(2).presence
    val gradient = jit(Autodiff.grad((scores: Tensor2[Query, Presence, Float32]) => loss(sureOfTheLine(2).copy(presence = scores), oneLine)))(presence)
    assert(gradient.value.abs.sum.item > 0f)
