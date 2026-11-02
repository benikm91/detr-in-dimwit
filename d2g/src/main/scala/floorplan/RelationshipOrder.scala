package d2g.floorplan

import dimwit.*
import syntheticFloorplansDataset.FloorplanClass
import syntheticFloorplansDataset.FloorplanClasses
import syntheticFloorplansDataset.FloorplanRecordBatch
import syntheticFloorplansDataset.FloorplanRelationship
import syntheticFloorplansDataset.FloorplanRelationships

/** The order a floor plan's relationships are written down in, one at a time, once its nodes are. */
enum RelationshipOrder:

  /** Any order: a record's relationships are a set. */
  case Free

  /** Kind by kind — walls joined to walls, then windows in their walls, then doors in theirs, then
    * what borders the rooms — and in any order within a kind.
    */
  case ByKind

  /** The rank of a relationship by what it is and the class of the node it starts from, by which
    * [[syntheticFloorplansDataset.FloorplanRecordBatch.permuted]] lays a target out in this order.
    */
  def kindRank: Tensor2[FloorplanRelationships, FloorplanClasses, Float32] = this match
    case Free   => FloorplanRecordBatch.anyKind
    case ByKind =>
      val ranks = FloorplanRelationship.values.toSeq.map: related =>
        Tensor1(Axis[FloorplanClasses], VType[Float32]).fromArray(FloorplanClass.values.map(startsFrom => (related, startsFrom) match
          case (FloorplanRelationship.In, FloorplanClass.Window) => 1f
          case (FloorplanRelationship.In, _)                     => 2f
          case (FloorplanRelationship.Borders, _)                => 3f
          case _                                                 => 0f
        ))
      stack(ranks, Axis[FloorplanRelationships])

object RelationshipOrder:

  /** The order a run names on its command line. */
  def named(name: String): RelationshipOrder = name match
    case "free"    => Free
    case "by-kind" => ByKind
    case other     => sys.error(s"no relationship order named '$other': free, by-kind")
