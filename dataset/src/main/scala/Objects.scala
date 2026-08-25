package dataset

import dimwit.*
import dimwit.Conversions.given
import dimwit.tensor.Tensor4

import scala.language.implicitConversions

/** Axis over the [[RelationClass]] values an edge is classified into. */
trait RelationClasses derives Label

/** The relationships of a record, as the adjacency matrix of an object view classifies them.
  *
  * There is no "no relation" class: a pair of objects carries each class or it does not,
  * independently of the others, so an unrelated pair is simply zero everywhere.
  */
enum RelationClass(val id: Int, val edgeClass: EdgeClass):
  case Connected extends RelationClass(0, EdgeClass.Connected)
  case Annotates extends RelationClass(1, EdgeClass.Annotates)

object RelationClass:

  def fromId(id: Int): RelationClass =
    values.find(_.id == id).getOrElse(throw IllegalArgumentException(s"unknown relation class id: $id"))

/** [[NoObject]] is DETR's "no object" class and marks an unused query slot. */
enum ObjectClass(val id: Int):
  case NoObject extends ObjectClass(0)
  case PartLine extends ObjectClass(1)
  case Text extends ObjectClass(2)

object ObjectClass:

  def fromId(id: Int): ObjectClass =
    values.find(_.id == id).getOrElse(throw IllegalArgumentException(s"unknown object class id: $id"))

  def of(nodeClass: NodeClass): ObjectClass = nodeClass match
    case NodeClass.Line       => PartLine
    case NodeClass.Annotation => Text
    case _                    => NoObject

/** The objects in an image: a [[Box]] per node, labelled with an [[ObjectClass.id]]. */
final case class Detection[Node, V](
    box: Box[Tuple1[Node], V],
    label: Tensor1[Node, Int32]
)

/** [[Detection]] for a batch of images along the axis `S`. */
final case class DetectionBatch[S, Node, V](
    box: Box[(S, Node), V],
    label: Tensor2[S, Node, Int32]
)

/** A record as it is drawn: a box around every drawn node, and the relationships between them as
  * an adjacency matrix over the positions those boxes sit in.
  *
  * `relations(i, j, k)` is `1` where the object at `i` carries the `k`-th [[RelationClass]]
  * towards the object at `j`. A symmetric relation is held both ways round here, unlike in the
  * record, where it is one node.
  */
final case class Objects[Node](
    detection: Detection[Node, Float32],
    relations: Tensor3[Node, Prime[Node], RelationClasses, Float32]
)

/** [[Objects]] for a batch of drawings along the axis `S`. */
final case class ObjectBatch[S, Node](
    detection: DetectionBatch[S, Node, Float32],
    relations: Tensor4[S, Node, Prime[Node], RelationClasses, Float32]
)

object Objects:

  /** The extent a box is widened to where a line is degenerate, in fractions of the canvas. */
  private val MinimumSize = 4f / Canvas

  /** The extent of the box around an annotation, which is a point. */
  private val AnnotationSize = 12f / Canvas

  /** The objects a record is drawn as. Going the other way is [[record]]. */
  def of[Node: Label, Edge: Label](record: Record[Node, Edge]): Objects[Node] =
    Objects(boxes(record), adjacency(record))

  def of[S: Label, Node: Label, Edge: Label](records: RecordBatch[S, Node, Edge]): ObjectBatch[S, Node] =
    val drawn = zipvmap(Axis[S])(records.nodeClass, records.xs, records.ys, records.edgeClass, records.links):
      case (nodeClass, xs, ys, edgeClass, links) =>
        val objects = of(Record(nodeClass, xs, ys, edgeClass, links))
        (
          centerX = objects.detection.box.centerX,
          centerY = objects.detection.box.centerY,
          width = objects.detection.box.width,
          height = objects.detection.box.height,
          label = objects.detection.label,
          relations = objects.relations
        )
    ObjectBatch(
      DetectionBatch(Box(drawn.centerX, drawn.centerY, drawn.width, drawn.height), drawn.label),
      drawn.relations
    )

  /** The record a drawing's objects came from.
    *
    * A line is axis aligned, so the long side of its box is the line and the short side is what
    * [[of]] widened it to; an annotation is the centre of its box. Packing an adjacency matrix
    * back into slots is a compaction rather than an elementwise map, so this reads the objects to
    * the host, which is where it is wanted — nothing scores on the device.
    */
  def record[Node: Label, Edge: Label](objects: Objects[Node], edges: AxisExtent[Edge]): Record[Node, Edge] =
    RecordGraph.of(objects).record(objects.detection.label.shape.extent(Axis[Node]), edges)

  private def boxes[Node: Label, Edge: Label](record: Record[Node, Edge]): Detection[Node, Float32] =
    val slots = record.nodeClass.shape.extent(Axis[Node])
    def point(index: Int) = (record.xs.slice(Axis[NodePoint].at(index)), record.ys.slice(Axis[NodePoint].at(index)))
    val ((startX, startY), (endX, endY)) = (point(0), point(1))
    val isLine = holds(record.nodeClass, NodeClass.Line.id)
    val isAnnotation = holds(record.nodeClass, NodeClass.Annotation.id)
    val drawn = isLine.asFloat(VType[Float32]) + isAnnotation.asFloat(VType[Float32])
    val annotationSize = Tensor1(slots).fill(AnnotationSize)
    def span(from: Tensor1[Node, Float32], to: Tensor1[Node, Float32]) =
      maximum((to - from).abs, Tensor1(slots).fill(MinimumSize))
    def labelled(objectClass: ObjectClass) = Tensor1(slots, VType[Int32]).fill(objectClass.id)
    Detection(
      box = Box(
        centerX = where(isLine, (startX + endX) *! 0.5f, startX) * drawn,
        centerY = where(isLine, (startY + endY) *! 0.5f, startY) * drawn,
        width = where(isLine, span(startX, endX), annotationSize) * drawn,
        height = where(isLine, span(startY, endY), annotationSize) * drawn
      ),
      label = where(isLine, labelled(ObjectClass.PartLine), where(isAnnotation, labelled(ObjectClass.Text), labelled(ObjectClass.NoObject)))
    )

  private def adjacency[Node: Label, Edge: Label](record: Record[Node, Edge]): Tensor3[Node, Prime[Node], RelationClasses, Float32] =
    val nodes = record.nodeClass.shape.extent(Axis[Node])
    val linkedNodes = Axis[Prime[Node]] -> nodes.size
    def linked(index: Int) = record.links.slice(Axis[NodeLink].at(index))
    val (subject, obj) = (linked(0), linked(1))

    def linking(relation: RelationClass, from: Tensor1[Edge, Int32], to: Tensor1[Edge, Int32]) =
      val edges = record.edgeClass.shape.extent(Axis[Edge])
      val present = holds(record.edgeClass, relation.edgeClass.id).asFloat(VType[Float32]).broadcastTo(Shape2(edges, nodes))
      (at(from, nodes) * present).dot(Axis[Edge])(at(to, linkedNodes))

    stack(
      RelationClass.values.map: relation =>
        val forward = linking(relation, subject, obj)
        if relation.edgeClass.isSymmetric then forward + linking(relation, obj, subject) else forward
      .toSeq,
      Axis[RelationClasses]
    ).transpose((Axis[Node], Axis[Prime[Node]], Axis[RelationClasses]))

  /** A one wherever a node is the one linked, over the axis it is linked in. */
  private def at[Named: Label, Over: Label](named: Tensor1[Named, Int32], over: AxisExtent[Over]): Tensor2[Named, Over, Float32] =
    val pairs = Shape2(named.shape.extent(Axis[Named]), over)
    Tensor1(over.axis, VType[Int32])
      .fromArray(Array.range(0, over.size))
      .broadcastTo(pairs)
      .elementEquals(named.broadcastTo(pairs))
      .asFloat(VType[Float32])

  private def holds[L: Label](classes: Tensor1[L, Int32], id: Int): Tensor1[L, Bool] =
    classes.elementEquals(Tensor.like(classes).fill(id))
