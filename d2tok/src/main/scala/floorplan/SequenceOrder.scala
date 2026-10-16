package d2tok.floorplan

import d2tok.*
import dimwit.*
import syntheticFloorplansDataset.FloorplanClass
import syntheticFloorplansDataset.FloorplanClasses
import syntheticFloorplansDataset.FloorplanRecordBatch

import scala.language.implicitConversions

/** The order a floor plan's nodes are written in. A sequence model learns to write one node after
  * the other, so the order of its targets is the order it learns.
  */
enum SequenceOrder(val name: String):

  /** A fresh random order every time a record is seen, as for a collection with no order of its own. */
  case Random extends SequenceOrder("random")

  /** The order the dataset stores a record's nodes in. */
  case Kept extends SequenceOrder("kept")

  /** Class by class — outside walls, inside walls, doors, outside doors, then windows — and in a
    * fresh random order within a class.
    */
  case ByClass extends SequenceOrder("class")

  /** Top to bottom and left to right by the pixel a node starts on, then by its class and the pixel
    * it ends on: one order for every record, none of it drawn at random.
    */
  case ByPosition extends SequenceOrder("position")

  /** `records` with their nodes laid out in this order, the positions they do not reach last.
    *
    * @param key  What a random order is drawn from.
    * @param canvas The side of the drawings in pixels, which a node's position is read on.
    */
  def laidOut[S: Label](records: FloorplanRecordBatch[S, Node, Relationship], key: Key, canvas: Int): FloorplanRecordBatch[S, Node, Relationship] =
    val (nodes, relationships) = (records.nodeClass.shape.extent(Axis[Node]), records.relationship.shape.extent(Axis[Relationship]))
    this match
      case Random     => records.permuted(key, nodes, relationships)
      case Kept       => records
      case ByClass    => records.permuted(key, nodes, relationships, SequenceOrder.classRank)
      case ByPosition => SequenceOrder.byPosition(records, canvas)

object SequenceOrder:
  def named(name: String): SequenceOrder = values.find(_.name == name).getOrElse(sys.error(s"no order named '$name': ${values.map(_.name).mkString(", ")}"))

  /** Where every class comes in [[ByClass]]: the classes not listed, [[FloorplanClass.NoNode]] among
    * them, after the ones that are.
    */
  private def classRank: Tensor1[FloorplanClasses, Float32] =
    val classes = Seq(FloorplanClass.OutsideWall, FloorplanClass.InsideWall, FloorplanClass.Door, FloorplanClass.OutsideDoor, FloorplanClass.Window)
    Tensor1(Axis[FloorplanClasses], VType[Float32]).fromArray(FloorplanClass.values.map(nodeClass => classes.indexOf(nodeClass) match
      case -1 => classes.size.toFloat
      case at => at.toFloat
    ))

  private def byPosition[S: Label](records: FloorplanRecordBatch[S, Node, Relationship], canvas: Int): FloorplanRecordBatch[S, Node, Relationship] =
    val classes = FloorplanClass.values.length
    val (nodeClass, startX, startY, endX, endY) =
      zipvmap(Axis[S])(records.nodeClass, records.startX, records.startY, records.endX, records.endY):
        case (nodeClass, startX, startY, endX, endY) =>
          def pixel(x: Tensor1[Node, Float32], y: Tensor1[Node, Float32]) = Pixels.of(y, canvas) *! canvas + Pixels.of(x, canvas)
          val starts = where(nodeClass elementEquals_! FloorplanClass.NoNode.id, Tensor.like(nodeClass).fill(canvas * canvas * classes), pixel(startX, startY) *! classes + nodeClass)
          // Least telling first: every sort keeps the order of what it finds equal.
          val order = Seq(pixel(endX, endY), starts).foldLeft(Tensor1(Axis[Node]).fromRange(0 until nodeClass.shape(Axis[Node]))):
            (order, key) => order.slice(Axis[Node].at(key.slice(Axis[Node].at(order)).argsort(Axis[Node])))
          def moved[W](values: Tensor1[Node, W]) = values.slice(Axis[Node].at(order))
          (moved(nodeClass), moved(startX), moved(startY), moved(endX), moved(endY))
    records.copy(nodeClass = nodeClass, startX = startX, startY = startY, endX = endX, endY = endY)
