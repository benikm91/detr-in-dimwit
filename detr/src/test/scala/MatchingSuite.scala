package detr.train

import detr.*
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
    for columns <- 1 to 4; rows <- columns to columns + 2; _ <- 1 to 3 do
      val values = Array.fill(rows, columns)(random.nextInt(100).toFloat)
      val matched = Matching.optimal(cost(values)).toArray.toSeq
      def total(rowOfColumn: Seq[Int]) = rowOfColumn.zipWithIndex.map((row, column) => values(row)(column)).sum
      assertEquals(matched.toSet.size, columns, s"rows assigned twice: $matched")
      assertEquals(total(matched), (0 until rows).permutations.map(_.take(columns)).map(total).min)

  test("gives up the cheapest pair when that makes the whole cheaper"):
    val values = Array(Array(1f, 2f), Array(2f, 100f))
    assertEquals(Matching.optimal(cost(values)).toArray.toSeq, Seq(1, 0))

  test("leaves over the rows no column needs"):
    val values = Array(Array(5f), Array(1f), Array(3f))
    assertEquals(Matching.optimal(cost(values)).toArray.toSeq, Seq(1))

  test("traces, so it can run inside jit and vmap"):
    val values = Array.tabulate(6, 4)((row, column) => (row * 4 + column) % 7 + 0.5f)
    val eager = Matching.optimal(cost(values)).toArray.toSeq
    assertEquals(jit(Matching.optimal[Row, Column, Float32])(cost(values)).toArray.toSeq, eager)
    val batched = stack(Seq(cost(values), cost(values)), Axis[Sample])
      .vmap(Axis[Sample])(Matching.optimal[Row, Column, Float32])
    assertEquals(batched.slice(Axis[Sample].at(1)).toArray.toSeq, eager)
