import d2s.model.jointSequenceMask
import dimwit.*

class JointSequenceMaskSuite extends munit.FunSuite:

  override def beforeAll(): Unit = dimwit.initialize()

  private trait Position derives Label

  test("a taken slot reads the taken ones up to itself, a prediction the taken ones before its slot and itself"):
    val (slots, queries) = (3, 2)
    val positions = Tensor1(Axis[Position], VType[Int32]).fromArray(Array.range(0, slots * (1 + queries)))
    val mask = jointSequenceMask[Position, Position](slots)(positions, positions).toArray
    def slotOf(at: Int) = if at < slots then at else (at - slots) % slots
    for target <- positions.toArray; source <- positions.toArray do
      val expected =
        if target < slots then source <= target
        else source == target || (source < slots && source < slotOf(target))
      assertEquals(mask(target)(source), expected, s"target $target reading source $source")
