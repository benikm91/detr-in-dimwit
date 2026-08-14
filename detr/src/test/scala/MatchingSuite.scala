import dimwit.*
import munit.FunSuite

class MatchingSuite extends FunSuite:

  trait Row derives Label
  trait Column derives Label
  trait Sample derives Label

  override def beforeAll(): Unit = dimwit.initialize()

  private def cost(values: Array[Array[Float]]): Tensor2[Row, Column, Float32] =
    Tensor2(Axis[Row], Axis[Column], VType[Float32]).fromArray(values)

  test("assigns every row a distinct column"):
    val random = scala.util.Random(1)
    for size <- 1 to 8; _ <- 1 to 5 do
      val matched = Matching.greedy(cost(Array.fill(size, size)(random.nextInt(100).toFloat))).toArray
      assertEquals(matched.toSet.size, size, s"columns assigned twice: ${matched.toSeq}")
      assert(matched.forall(column => column >= 0 && column < size))

  test("takes the cheapest pairs first"):
    // Every row has one clearly cheapest column; greedy and optimal agree here.
    val values = Array.tabulate(6, 6)((row, column) => if column == (row + 2) % 6 then 0f else 10f + row)
    assertEquals(Matching.greedy(cost(values)).toArray.toSeq, (0 until 6).map(row => (row + 2) % 6))

  test("stays a permutation when some columns are far more expensive"):
    // The loss puts a surcharge on padding slots, so open and taken pairs differ wildly in scale.
    val values = Array.tabulate(6, 6)((row, column) => if column >= 4 then 1e3f else (row * 3 + column) % 5 + 0.5f)
    assertEquals(Matching.greedy(cost(values)).toArray.toSet.size, 6)

  test("traces, so it can run inside jit and vmap"):
    val values = Array.tabulate(4, 4)((row, column) => (row * 4 + column) % 7 + 0.5f)
    val eager = Matching.greedy(cost(values)).toArray.toSeq
    assertEquals(jit(Matching.greedy[Row, Column, Float32])(cost(values)).toArray.toSeq, eager)
    val batched = stack(Seq(cost(values), cost(values)), Axis[Sample])
      .vmap(Axis[Sample])(Matching.greedy[Row, Column, Float32])
    assertEquals(batched.slice(Axis[Sample].at(1)).toArray.toSeq, eager)
