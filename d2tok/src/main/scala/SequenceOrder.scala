package d2tok

import dataset.NodeClass
import dataset.NodeClasses
import dataset.RecordBatch
import dimwit.*

import scala.language.implicitConversions

/** The order a record's nodes are written in. A sequence model learns to write one node after the
  * other, so the order of its targets is the order it learns.
  */
enum SequenceOrder(val name: String):

  /** A fresh random order every time a record is seen, as for a collection with no order of its own. */
  case Random extends SequenceOrder("random")

  /** The order the corpus stores a record's nodes in. */
  case Kept extends SequenceOrder("kept")

  /** Class by class — points, lines, circles, arcs, then annotations — and in a fresh random order
    * within a class.
    */
  case ByClass extends SequenceOrder("class")

  /** Top to bottom and left to right by the pixel a node starts on, then by its class, the pixel it
    * ends on and the one it bends through: one order for every record, none of it drawn at random.
    */
  case ByPosition extends SequenceOrder("position")

  /** `records` with their nodes laid out in this order, the positions they do not reach last.
    *
    * @param key  What a random order is drawn from.
    * @param canvas The side of the drawings in pixels, which a node's position is read on.
    */
  def laidOut[S: Label](records: RecordBatch[S, Node, Relationship], key: Key, canvas: Int): RecordBatch[S, Node, Relationship] =
    val (nodes, relationships) = (records.nodeClass.shape.extent(Axis[Node]), records.edgeClass.shape.extent(Axis[Relationship]))
    this match
      case Random     => records.permuted(key, nodes, relationships)
      case Kept       => records
      case ByClass    => records.permuted(key, nodes, relationships, SequenceOrder.classRank)
      case ByPosition => SequenceOrder.byPosition(records, canvas)

object SequenceOrder:
  def named(name: String): SequenceOrder = values.find(_.name == name).getOrElse(sys.error(s"no order named '$name': ${values.map(_.name).mkString(", ")}"))

  /** Where every class comes in [[ByClass]]: the classes not listed, [[NodeClass.NoNode]] among them,
    * after the ones that are.
    */
  private def classRank: Tensor1[NodeClasses, Float32] =
    val classes = Seq(NodeClass.Point, NodeClass.Line, NodeClass.Circle, NodeClass.Arc, NodeClass.Annotation)
    Tensor1(Axis[NodeClasses], VType[Float32]).fromArray(NodeClass.values.map(nodeClass => classes.indexOf(nodeClass) match
      case -1 => classes.size.toFloat
      case at => at.toFloat
    ))

  private def byPosition[S: Label](records: RecordBatch[S, Node, Relationship], canvas: Int): RecordBatch[S, Node, Relationship] =
    val classes = NodeClass.values.length
    val (nodeClass, construction, startX, startY, endX, endY, midX, midY) =
      zipvmap(Axis[S])(records.nodeClass, records.construction, records.startX, records.startY, records.endX, records.endY, records.midX, records.midY):
        case (nodeClass, construction, startX, startY, endX, endY, midX, midY) =>
          def pixel(x: Tensor1[Node, Float32], y: Tensor1[Node, Float32]) = Pixels.of(y, canvas) *! canvas + Pixels.of(x, canvas)
          val starts = where(nodeClass elementEquals_! NodeClass.NoNode.id, Tensor.like(nodeClass).fill(canvas * canvas * classes), pixel(startX, startY) *! classes + nodeClass)
          // Least telling first: every sort keeps the order of what it finds equal.
          val order = Seq(pixel(midX, midY), pixel(endX, endY), starts).foldLeft(Tensor1(Axis[Node]).fromRange(0 until nodeClass.shape(Axis[Node]))):
            (order, key) => order.slice(Axis[Node].at(key.slice(Axis[Node].at(order)).argsort(Axis[Node])))
          def moved[W](values: Tensor1[Node, W]) = values.slice(Axis[Node].at(order))
          (moved(nodeClass), moved(construction), moved(startX), moved(startY), moved(endX), moved(endY), moved(midX), moved(midY))
    records.copy(nodeClass = nodeClass, construction = construction, startX = startX, startY = startY, endX = endX, endY = endY, midX = midX, midY = midY)
