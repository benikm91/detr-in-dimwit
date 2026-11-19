package d2g.sketch

import dataset.EdgeClass
import dataset.EdgeClasses
import dimwit.*

/** The order a drawing's relationships are written down in, one at a time, once its nodes are. */
enum RelationshipOrder:

  /** Any order: a record's relationships are a set. */
  case Free

  /** Class by class — lines connected in their corners, then annotations to their lines — and in
    * any order within a class.
    */
  case ByKind

  /** The rank of every class, by which [[dataset.RecordBatch.permuted]] lays a target out in this
    * order: [[EdgeClass.NoEdge]] after every other.
    */
  def classRank: Tensor1[EdgeClasses, Float32] = this match
    case Free   => EdgeClass.indicator(VType[Float32])(!_.isEdge)
    case ByKind =>
      Tensor1(Axis[EdgeClasses], VType[Float32]).fromArray(EdgeClass.values.map:
        case EdgeClass.Connected => 0f
        case EdgeClass.Annotates => 1f
        case EdgeClass.NoEdge    => 2f
      )

object RelationshipOrder:

  /** The order a run names on its command line. */
  def named(name: String): RelationshipOrder = name match
    case "free"    => Free
    case "by-kind" => ByKind
    case other     => sys.error(s"no relationship order named '$other': free, by-kind")
