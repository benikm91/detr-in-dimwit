package detr.train

import detr.*
import detr.model.NodeHead.NodeLogits
import dataset.NodeClass
import dataset.NodeClasses
import dataset.IsConstruction
import dataset.RecordNodes
import dimwit.*
import munit.FunSuite

class HungarianLossSuite extends FunSuite:

  override def beforeAll(): Unit = dimwit.initialize()

  private val canvas = 8
  private val queries = 4
  private val loss = HungarianLoss(VType[Float32], canvas)

  /** A record of one line, from (2, 4) to (6, 4) in pixels, and one slot holding no node. */
  private val oneLine = RecordNodes(
    nodeClass = Tensor1(Axis[Node], VType[Int32]).fromArray(Array(NodeClass.Line.id, NodeClass.NoNode.id)),
    construction = Tensor1(Axis[Node], VType[Int32]).fromArray(Array(0, 0)),
    startX = Tensor1(Axis[Node], VType[Float32]).fromArray(Array(0.25f, 0f)),
    startY = Tensor1(Axis[Node], VType[Float32]).fromArray(Array(0.5f, 0f)),
    endX = Tensor1(Axis[Node], VType[Float32]).fromArray(Array(0.75f, 0f)),
    endY = Tensor1(Axis[Node], VType[Float32]).fromArray(Array(0.5f, 0f)),
    midX = Tensor1(Axis[Node], VType[Float32]).fromArray(Array(0f, 0f)),
    midY = Tensor1(Axis[Node], VType[Float32]).fromArray(Array(0f, 0f))
  )

  /** Every query sure of no node, except those `answering`, which are sure of that line. */
  private def sureOfTheLine(answering: Int*): NodeLogits[Float32] =
    def sure[L: Label](axis: Axis[L], size: Int)(answer: Int => Int) =
      Tensor2(Axis[Query], axis, VType[Float32]).fromArray(Array.tabulate(queries, size)((query, value) => if value == answer(query) then 20f else 0f))
    def pixel(at: Int) = sure(Axis[Pixel], canvas)(query => if answering.contains(query) then at else 0)
    NodeLogits(
      nodeClass = sure(Axis[NodeClasses], NodeClass.values.length)(query => if answering.contains(query) then NodeClass.Line.id else NodeClass.NoNode.id),
      construction = sure(Axis[IsConstruction], 2)(_ => 0),
      startX = pixel(2),
      startY = pixel(4),
      endX = pixel(6),
      endY = pixel(4),
      midX = pixel(0),
      midY = pixel(0)
    )

  test("costs next to nothing when a query answers every node and the rest answer none"):
    assert(loss(sureOfTheLine(2), oneLine).item < 1e-3f)

  test("does not care which query answers"):
    assertEqualsFloat(loss(sureOfTheLine(0), oneLine).item, loss(sureOfTheLine(3), oneLine).item, 1e-6f)

  test("charges a query that answers where none is asked for"):
    assert(loss(sureOfTheLine(1, 2), oneLine).item > 1f)

  test("has a gradient inside jit"):
    val classScores = sureOfTheLine(2).nodeClass
    val gradient = jit(Autodiff.grad((scores: Tensor2[Query, NodeClasses, Float32]) => loss(sureOfTheLine(2).copy(nodeClass = scores), oneLine)))(classScores)
    assert(gradient.value.abs.sum.item > 0f)
