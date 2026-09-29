package detr.train

import detr.*
import detr.model.*
import detr.eval.*
import detr.config.*
import dimwit.*
import munit.FunSuite

class MatchingSuite extends FunSuite:

  trait Row derives Label
  trait Column derives Label
  trait Sample derives Label

  override def beforeAll(): Unit = dimwit.initialize()

  private def cost(values: Array[Array[Float]]): Tensor2[Row, Column, Float32] =
    Tensor2(Axis[Row], Axis[Column], VType[Float32]).fromArray(values)

  test("costs no more than any other assignment"):
    val random = scala.util.Random(1)
    for size <- 1 to 6; _ <- 1 to 5 do
      val values = Array.fill(size, size)(random.nextInt(100).toFloat)
      val matched = Matching.optimal(cost(values)).toArray
      def total(assignment: Seq[Int]) = assignment.zipWithIndex.map((column, row) => values(row)(column)).sum
      assertEquals(matched.toSet.size, size, s"columns assigned twice: ${matched.toSeq}")
      assertEquals(total(matched.toSeq), (0 until size).permutations.map(total).min)

  test("gives up the cheapest pair when that makes the whole cheaper"):
    val values = Array(Array(1f, 2f), Array(2f, 100f))
    assertEquals(Matching.optimal(cost(values)).toArray.toSeq, Seq(1, 0))

  test("traces, so it can run inside jit and vmap"):
    val values = Array.tabulate(4, 4)((row, column) => (row * 4 + column) % 7 + 0.5f)
    val eager = Matching.optimal(cost(values)).toArray.toSeq
    assertEquals(jit(Matching.optimal[Row, Column, Float32])(cost(values)).toArray.toSeq, eager)
    val batched = stack(Seq(cost(values), cost(values)), Axis[Sample])
      .vmap(Axis[Sample])(Matching.optimal[Row, Column, Float32])
    assertEquals(batched.slice(Axis[Sample].at(1)).toArray.toSeq, eager)
