package syntheticFloorplansDataset

import dataset.Point

class FloorplanScoringSuite extends munit.FunSuite:

  private def wall(x0: Float, y0: Float, x1: Float, y1: Float) = FloorplanNode(FloorplanClass.InsideWall, Point(x0, y0), Point(x1, y1))
  private val bedroom = FloorplanNode(FloorplanClass.Bedroom, Point(0f, 0f), Point(0f, 0f))

  test("a room is found as the room of its class bordered by the same walls, whatever order the rooms are written in"):
    val (left, right) = (wall(0.1f, 0.1f, 0.1f, 0.9f), wall(0.9f, 0.1f, 0.9f, 0.9f))
    val target = FloorplanGraph(
      Seq(left, right, bedroom, bedroom),
      Seq(FloorplanEdge(FloorplanRelationship.Borders, 0, 2), FloorplanEdge(FloorplanRelationship.Borders, 1, 3))
    )
    val writtenTheOtherWay = FloorplanGraph(
      Seq(right, left, bedroom, bedroom),
      Seq(FloorplanEdge(FloorplanRelationship.Borders, 1, 3), FloorplanEdge(FloorplanRelationship.Borders, 0, 2))
    )
    assertEquals(FloorplanScoring.matching(target, writtenTheOtherWay, 0.01f), Map(0 -> 1, 1 -> 0, 2 -> 3, 3 -> 2))
    assert(FloorplanScoring.score(target, writtenTheOtherWay, 0.01f).isExact)
