package d2tok.floorplangraph

import d2tok.*
import dimwit.*
import syntheticFloorplansDataset.FloorplanRecordBatch
import syntheticFloorplansDataset.FloorplanRelationship

import scala.language.implicitConversions

/** The order a floor plan's relationships are written in, once its nodes have been laid out. */
enum RelationshipOrder(val name: String):

  /** A fresh random order every time a record is seen. */
  case Random extends RelationshipOrder("random")

  /** By the numbers of the nodes they link, as the nodes are written: by the first, then by the
    * second. A joined pair names its lower number first.
    */
  case ByNodes extends RelationshipOrder("by-nodes")

  /** `records` with their relationships laid out in this order, the positions they do not reach last.
    *
    * @param key What a random order is drawn from.
    */
  def laidOut[S: Label](records: FloorplanRecordBatch[S, Node, Relationship], key: Key): FloorplanRecordBatch[S, Node, Relationship] =
    val nodes = records.nodeClass.shape(Axis[Node])
    val keys = key.splitToTensor(records.relationship.shape.extent(Axis[S]))
    val (relationship, subject, obj) =
      zipvmap(Axis[S])(records.relationship, records.subject, records.obj, keys):
        case (relationship, subject, obj, key) =>
          val slots = relationship.shape.extent(Axis[Relationship])
          val rank = this match
            case Random  => dimwit.Random.permutation(slots)(key.item)
            case ByNodes => subject *! nodes + obj
          val absent = relationship elementEquals_! FloorplanRelationship.NoRelationship.id
          val order = where(absent, Tensor.like(rank).fill(nodes * nodes + slots.size), rank).argsort(Axis[Relationship])
          def moved(values: Tensor1[Relationship, Int32]) = values.slice(Axis[Relationship].at(order))
          (moved(relationship), moved(subject), moved(obj))
    records.copy(relationship = relationship, subject = subject, obj = obj)

object RelationshipOrder:
  def named(name: String): RelationshipOrder = values.find(_.name == name).getOrElse(sys.error(s"no order named '$name': ${values.map(_.name).mkString(", ")}"))
