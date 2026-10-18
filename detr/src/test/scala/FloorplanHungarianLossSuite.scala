package detr.floorplan.train

import detr.*
import detr.floorplan.model.NodeHead.NodeLogits
import dimwit.*
import munit.FunSuite
import syntheticFloorplansDataset.FloorplanClass
import syntheticFloorplansDataset.FloorplanClasses
import syntheticFloorplansDataset.FloorplanNodes

class FloorplanHungarianLossSuite extends FunSuite:

  override def beforeAll(): Unit = dimwit.initialize()

  private val canvas = 8
  private val queries = 4
  private val loss = HungarianLoss(VType[Float32], canvas)

  /** A plan of one wall, from (2, 4) to (6, 4) in pixels, and one slot holding no node. */
  private val oneWall = FloorplanNodes(
    nodeClass = Tensor1(Axis[Node], VType[Int32]).fromArray(Array(FloorplanClass.OutsideWall.id, FloorplanClass.NoNode.id)),
    startX = Tensor1(Axis[Node], VType[Float32]).fromArray(Array(0.25f, 0f)),
    startY = Tensor1(Axis[Node], VType[Float32]).fromArray(Array(0.5f, 0f)),
    endX = Tensor1(Axis[Node], VType[Float32]).fromArray(Array(0.75f, 0f)),
    endY = Tensor1(Axis[Node], VType[Float32]).fromArray(Array(0.5f, 0f))
  )

  /** Every query sure of no node, except those `answering`, which are sure of that wall. */
  private def sureOfTheWall(answering: Int*): NodeLogits[Float32] =
    def sure[L: Label](axis: Axis[L], size: Int)(answer: Int => Int) =
      Tensor2(Axis[Query], axis, VType[Float32]).fromArray(Array.tabulate(queries, size)((query, value) => if value == answer(query) then 20f else 0f))
    def pixel(at: Int) = sure(Axis[Pixel], canvas)(query => if answering.contains(query) then at else 0)
    NodeLogits(
      nodeClass = sure(Axis[FloorplanClasses], FloorplanClass.values.length)(query => if answering.contains(query) then FloorplanClass.OutsideWall.id else FloorplanClass.NoNode.id),
      startX = pixel(2),
      startY = pixel(4),
      endX = pixel(6),
      endY = pixel(4)
    )

  test("costs next to nothing when a query answers every node and the rest answer none"):
    assert(loss(sureOfTheWall(2), oneWall).item < 1e-3f)

  test("does not care which query answers"):
    assertEqualsFloat(loss(sureOfTheWall(0), oneWall).item, loss(sureOfTheWall(3), oneWall).item, 1e-6f)

  test("charges a query that answers where none is asked for"):
    assert(loss(sureOfTheWall(1, 2), oneWall).item > 1f)
