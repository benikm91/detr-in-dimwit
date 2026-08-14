import dimwit.*
import munit.FunSuite

class HungarianMatchingSuite extends FunSuite:

  trait Row derives Label
  trait Column derives Label

  override def beforeAll(): Unit = dimwit.initialize()

  private def optimalCost(values: Array[Array[Float]], rows: Int, columns: Int): Float =
    (0 until columns).toList
      .combinations(rows)
      .flatMap(_.permutations)
      .map(assignment => assignment.zipWithIndex.map((column, row) => values(row)(column)).sum)
      .min

  private def check(rows: Int, columns: Int, random: scala.util.Random): Unit =
    val values = Array.fill(rows, columns)(random.nextInt(100).toFloat)
    val matched = HungarianMatching(Tensor2(Axis[Row], Axis[Column], VType[Float32]).fromArray(values)).toArray
    assertEquals(matched.length, rows)
    assertEquals(matched.toSet.size, rows, s"columns assigned twice: ${matched.toSeq}")
    val total = matched.zipWithIndex.map((column, row) => values(row)(column)).sum
    assertEqualsFloat(total, optimalCost(values, rows, columns), 1e-3f, s"suboptimal for ${values.map(_.toSeq).toSeq}")

  test("square assignments are optimal"):
    val random = scala.util.Random(1)
    for size <- 1 to 6; _ <- 1 to 5 do check(size, size, random)

  test("rectangular assignments are optimal"):
    val random = scala.util.Random(2)
    for rows <- 1 to 4; extra <- 1 to 3; _ <- 1 to 5 do check(rows, rows + extra, random)
