package d2g.floorplan

import dimwit.*
import syntheticFloorplansDataset.FloorplanClass
import syntheticFloorplansDataset.FloorplanClasses

/** The order a floor plan's relationships are written down in, one at a time, once its nodes are. */
enum RelationshipOrder:

  /** Any order: a record's relationships are a set. */
  case Free

  /** Kind by kind — walls joined to walls, then windows in their walls, then doors in theirs — and
    * in any order within a kind.
    */
  case ByKind

  /** The rank of a relationship by the class of the node it starts from, by which
    * [[syntheticFloorplansDataset.FloorplanRecordBatch.permuted]] lays a target out in this order.
    * That class tells the kinds apart: a joined pair starts from a wall, a relationship `in` a wall
    * from the opening.
    */
  def startRank: Tensor1[FloorplanClasses, Float32] = this match
    case Free   => FloorplanClass.indicator(VType[Float32])(_ => false)
    case ByKind =>
      Tensor1(Axis[FloorplanClasses], VType[Float32]).fromArray(FloorplanClass.values.map:
        case FloorplanClass.Window                           => 1f
        case FloorplanClass.Door | FloorplanClass.OutsideDoor => 2f
        case _                                               => 0f
      )

object RelationshipOrder:

  /** The order a run names on its command line. */
  def named(name: String): RelationshipOrder = name match
    case "free"    => Free
    case "by-kind" => ByKind
    case other     => sys.error(s"no relationship order named '$other': free, by-kind")
