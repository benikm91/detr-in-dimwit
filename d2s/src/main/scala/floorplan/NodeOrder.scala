package d2s.floorplan

import dimwit.*
import syntheticFloorplansDataset.FloorplanClass
import syntheticFloorplansDataset.FloorplanClasses

/** The order a floor plan's nodes are written down in, one at a time. */
enum NodeOrder:

  /** Any order: a record's nodes are a set. */
  case Free

  /** Class by class, in the order `classes` lists them, and in any order within a class. */
  case ByClass(classes: Seq[FloorplanClass])

  /** The rank of every class, by which [[syntheticFloorplansDataset.FloorplanRecordBatch.permuted]]
    * lays a target out in this order: the classes not listed, [[FloorplanClass.NoNode]] among them,
    * after the ones that are.
    */
  def classRank: Tensor1[FloorplanClasses, Float32] = this match
    case Free             => FloorplanClass.indicator(VType[Float32])(!_.isNode)
    case ByClass(classes) =>
      Tensor1(Axis[FloorplanClasses], VType[Float32]).fromArray(FloorplanClass.values.map(nodeClass => classes.indexOf(nodeClass) match
        case -1 => classes.size.toFloat
        case at => at.toFloat
      ))

object NodeOrder:

  private val wallsFirst = Seq(FloorplanClass.OutsideWall, FloorplanClass.InsideWall, FloorplanClass.OutsideDoor, FloorplanClass.Door, FloorplanClass.Window)

  /** The walls first, then what stands in them: an opening written later can be placed against the
    * wall it stands in.
    */
  val WallsFirst: NodeOrder = ByClass(wallsFirst)

  /** The same classes the other way round: whether it is this order of the classes that helps, or
    * any order of them.
    */
  val WallsLast: NodeOrder = ByClass(wallsFirst.reverse)

  /** The order a run names on its command line. */
  def named(name: String): NodeOrder = name match
    case "free"        => Free
    case "walls-first" => WallsFirst
    case "walls-last"  => WallsLast
    case other         => sys.error(s"no order named '$other': free, walls-first, walls-last")
