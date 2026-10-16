package syntheticFloorplansDataset

import dataset.DrawingDataset.Split
import dimwit.*
import munit.FunSuite

/** The synthetic floor plans, read from Hugging Face: what the records hold, and how the stream runs. */
class SyntheticFloorplansSuite extends FunSuite:

  override def beforeAll(): Unit = dimwit.initialize()

  /** The validation split is downloaded the first time it is opened. */
  override def munitTimeout: scala.concurrent.duration.Duration = scala.concurrent.duration.Duration(20, "min")

  private trait Width derives Label
  private trait Height derives Label
  private trait Channel derives Label
  private trait Node derives Label
  private trait Edge derives Label
  private trait Batch derives Label

  private lazy val validation = SyntheticFloorplans.open(Axis[Width], Axis[Height], Axis[Channel], Axis[Node], Axis[Edge])(Split.Validation)

  test("the training split is its nine parts in seed order, and validation one part"):
    val train = SyntheticFloorplans.open(Axis[Width], Axis[Height], Axis[Channel], Axis[Node], Axis[Edge])(Split.Train)
    assertEquals(train.parts, (0 until 9).map(part => (s"train-0$part", 50_000)))
    assertEquals(validation.parts, Seq(("val", 5000)))

  test("a record holds horizontal and vertical segments, openings in the wall they belong in, walls joined at a shared end"):
    for sample <- validation.samples.take(40) do
      val record = sample.target
      val classes = record.nodeClass.toArray.map(FloorplanClass.fromId)
      val drawn = classes.indices.filter(classes(_).isNode)
      assert(drawn.nonEmpty && drawn == classes.indices.take(drawn.size), "the nodes come first, the padding after")
      val (startX, startY, endX, endY) = (record.startX.toArray, record.startY.toArray, record.endX.toArray, record.endY.toArray)
      def ends(at: Int) = Set((startX(at), startY(at)), (endX(at), endY(at)))
      for at <- drawn do
        assert(Seq(startX(at), startY(at), endX(at), endY(at)).forall(v => v >= 0f && v <= 1f), s"node $at lies off the drawing")
        assert(startX(at) == endX(at) || startY(at) == endY(at), s"node $at is not horizontal or vertical")
        assert(startX(at) <= endX(at) && startY(at) <= endY(at), s"node $at is not read rightwards or downwards")
      val (relationship, subject, obj) = (record.relationship.toArray.map(FloorplanRelationship.fromId), record.subject.toArray, record.obj.toArray)
      val walls = Set(FloorplanClass.OutsideWall, FloorplanClass.InsideWall)
      for at <- relationship.indices if relationship(at).isRelationship do
        relationship(at) match
          case FloorplanRelationship.Joined =>
            assert(walls(classes(subject(at))) && walls(classes(obj(at))) && subject(at) < obj(at))
            assert((ends(subject(at)) & ends(obj(at))).nonEmpty, s"walls ${subject(at)} and ${obj(at)} share no end")
          case FloorplanRelationship.In =>
            val belongs = if classes(subject(at)) == FloorplanClass.Door then FloorplanClass.InsideWall else FloorplanClass.OutsideWall
            assertEquals(classes(obj(at)), belongs, s"a ${classes(subject(at))} in a ${classes(obj(at))}")
          case FloorplanRelationship.NoRelationship => ()
      val hosted = relationship.indices.filter(relationship(_) == FloorplanRelationship.In).map(subject).sorted
      assertEquals(hosted, drawn.filterNot(at => walls(classes(at))).sorted, "every opening stands in exactly one wall")

  test("the batches carry on where they left off, and wrap round at the end of the split"):
    val batch = Axis[Batch] -> 8
    def first(afterSteps: Int) = validation.batches(batch, afterSteps).next()
    val stream = validation.batches(batch, afterSteps = 0)
    val fourth = stream.drop(3).next()
    assertEquals(first(3).target.nodeClass.toArray.toSeq.map(_.toSeq), fourth.target.nodeClass.toArray.toSeq.map(_.toSeq))
    // 5,000 plans make 625 whole batches of 8: the 626th is the first again.
    assertEquals(first(625).target.startX.toArray.toSeq.map(_.toSeq), first(0).target.startX.toArray.toSeq.map(_.toSeq))
    assertEquals(fourth.pixels.shape(Axis[Width]), SyntheticFloorplans.canvas)
