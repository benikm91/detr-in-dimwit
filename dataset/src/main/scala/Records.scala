package dataset

import dimwit.*

import scala.util.Random

/** Axis over the [[NodeClass]] values a node of a record is classified into. */
trait NodeClasses derives Label

/** Axis of the points a node is placed by: a line has two, an annotation one. */
trait NodePoint derives Label

/** Axis of the nodes a relationship links: its subject, then its object. */
trait NodeLink derives Label

/** What a node of a record is.
  *
  * [[Line]] and [[Annotation]] are drawn; [[Connected]] and [[Annotates]] are the relationships
  * between them, which a record holds as nodes of their own so that a graph is a set. [[NoNode]]
  * marks a position the record does not reach, which is also where a transcription of it stops.
  */
enum NodeClass(val id: Int, val pointNames: Seq[String]):

  case NoNode extends NodeClass(0, Seq.empty)
  case Line extends NodeClass(1, Seq("start", "end"))
  case Annotation extends NodeClass(2, Seq("centre"))

  /** Two lines meeting in a corner, held once with the two it links in ascending order. */
  case Connected extends NodeClass(3, Seq.empty)
  case Annotates extends NodeClass(4, Seq.empty)

  def numPoints: Int = pointNames.length

  def numLinks: Int = if isRelationship then 2 else 0

  def isDrawn: Boolean = this == Line || this == Annotation

  def isRelationship: Boolean = this == Connected || this == Annotates

  def isSymmetric: Boolean = this == Connected

object NodeClass:

  def fromId(id: Int): NodeClass =
    values.find(_.id == id).getOrElse(throw IllegalArgumentException(s"unknown node class id: $id"))

  val maxPoints: Int = values.map(_.numPoints).max

  val maxLinks: Int = values.map(_.numLinks).max

  def indicator[V: IsFloating](vtype: VType[V])(holds: NodeClass => Boolean): Tensor1[NodeClasses, V] =
    Tensor1(Axis[NodeClasses], vtype).fromArray(values.map(nodeClass => if holds(nodeClass) then 1f else 0f))

  /** Which points a class places itself by, so that a node is only measured on what it carries. */
  def usedPoints[V: IsFloating](vtype: VType[V]): Tensor2[NodeClasses, NodePoint, V] =
    used(vtype, maxPoints, _.numPoints)

  /** Which nodes a class links, which is both of them for a relationship and neither otherwise. */
  def usedLinks[V: IsFloating](vtype: VType[V]): Tensor2[NodeClasses, NodeLink, V] =
    used(vtype, maxLinks, _.numLinks)

  private def used[Carried: Label, V: IsFloating](vtype: VType[V], width: Int, carries: NodeClass => Int): Tensor2[NodeClasses, Carried, V] =
    Tensor2(Axis[NodeClasses], Axis[Carried], vtype).fromArray(
      values.map(nodeClass => Array.tabulate(width)(carried => if carried < carries(nodeClass) then 1f else 0f))
    )

/** A point of the canvas, normalized to it. */
final case class Point(x: Float, y: Float)

/** What a drawing encodes: its nodes, and the relationships between them as nodes of their own.
  *
  * `xs` and `ys` place a node on the canvas; `links` name the nodes a relationship relates, by
  * their position along the `Node` axis. A node leaves at zero whatever its class does not carry —
  * a relationship has no points, a drawn node no links.
  */
final case class Record[Node](
    nodeClass: Tensor1[Node, Int32],
    xs: Tensor2[Node, NodePoint, Float32],
    ys: Tensor2[Node, NodePoint, Float32],
    links: Tensor2[Node, NodeLink, Int32]
)

/** [[Record]] for a batch of drawings along the axis `S`. */
final case class RecordBatch[S, Node](
    nodeClass: Tensor2[S, Node, Int32],
    xs: Tensor3[S, Node, NodePoint, Float32],
    ys: Tensor3[S, Node, NodePoint, Float32],
    links: Tensor3[S, Node, NodeLink, Int32]
):

  /** The same records, laid out again in `slots` positions with their nodes in a fresh random
    * order: the drawn nodes first, the relationships between them after them, and the positions
    * the record does not reach last.
    *
    * A record is a set, so the order it is written down in is the model's to be indifferent to,
    * which is what drawing a new one every step is for. This one is drawn on the device — nothing
    * is read back to lay it out again.
    */
  def permuted(key: Key, slots: AxisExtent[Node])(using Label[S], Label[Node]): RecordBatch[S, Node] =
    val drawings = nodeClass.shape.extent(Axis[S])
    val padded = paddedTo(slots, drawings)
    val keys = key.splitToTensor(drawings)
    val (permutedClasses, permutedXs, permutedYs, permutedLinks) =
      zipvmap(Axis[S])(padded.nodeClass, padded.xs, padded.ys, padded.links, keys):
        case (nodeClass, xs, ys, links, key) =>
          val permuted = RecordBatch.permutedRecord(Record(nodeClass, xs, ys, links), key.item)
          (permuted.nodeClass, permuted.xs, permuted.ys, permuted.links)
    RecordBatch(permutedClasses, permutedXs, permutedYs, permutedLinks)

  /** The same records in as many positions, the ones they do not reach holding no node. */
  private def paddedTo(slots: AxisExtent[Node], drawings: AxisExtent[S])(using Label[S], Label[Node]): RecordBatch[S, Node] =
    val held = nodeClass.shape(Axis[Node])
    require(held <= slots.size, s"records of $held positions do not fit in ${slots.size}")
    val empty = Axis[Node] -> (slots.size - held)
    RecordBatch(
      nodeClass = concatenate(nodeClass, Tensor(Shape(drawings, empty), VType[Int32]).fill(NodeClass.NoNode.id), Axis[Node]),
      xs = concatenate(xs, Tensor(Shape(drawings, empty, points), VType[Float32]).fill(0f), Axis[Node]),
      ys = concatenate(ys, Tensor(Shape(drawings, empty, points), VType[Float32]).fill(0f), Axis[Node]),
      links = concatenate(links, Tensor(Shape(drawings, empty, ends), VType[Int32]).fill(0), Axis[Node])
    )

  private def points = Axis[NodePoint] -> NodeClass.maxPoints

  private def ends = Axis[NodeLink] -> NodeClass.maxLinks

object RecordBatch:

  /** One record in a fresh random order, with the drawn nodes kept ahead of the relationships and
    * the positions it does not reach after both. A relationship names the nodes it links by their
    * position, so the names are read in the new order too.
    */
  private def permutedRecord[Node: Label](record: Record[Node], key: Key): Record[Node] =
    val slots = record.nodeClass.shape.extent(Axis[Node])
    val shuffle = dimwit.Random.permutation(slots)(key)
    def classIs(holds: NodeClass => Boolean)(of: Tensor1[Node, Int32]) =
      NodeClass.indicator(VType[Float32])(holds).take(Axis[NodeClasses])(of)

    val isRelationship = classIs(_.isRelationship)(record.nodeClass)
    val isEmpty = classIs(_ == NodeClass.NoNode)(record.nodeClass)
    val block = isRelationship + isEmpty + isEmpty // drawn nodes 0, relationships 1, empty positions 2
    val within = shuffle.asFloat(VType[Float32])
    val order = (block * Tensor.like(block).fill(slots.size.toFloat) + within).argsort(Axis[Node])
    val nodeClass = record.nodeClass.take(Axis[Node])(order)

    // A relationship names the nodes it links by their position, and a position that held the node
    // `at` before holds it at `inverseOrder(at)` now.
    val inverseOrder = order.argsort(Axis[Node])
    val moved = record.links.take(Axis[Node])(order).vmap(Axis[NodeLink])(inverseOrder.take(Axis[Node])(_))
    val ascending = stack(Seq(moved.min(Axis[NodeLink]), moved.max(Axis[NodeLink])), newAxis = Axis[NodeLink], afterAxis = Axis[Node])
    def is(holds: NodeClass => Boolean) = classIs(holds)(nodeClass) > Tensor.like(block).fill(0f)
    Record(
      nodeClass = nodeClass,
      xs = record.xs.take(Axis[Node])(order),
      ys = record.ys.take(Axis[Node])(order),
      // Two things [[RecordGraph.placed]] holds that a new order undoes: a symmetric relationship
      // names the two it links in ascending order, and a node that links nothing links position 0.
      links = where_!(is(_.isRelationship), where_!(is(_.isSymmetric), ascending, moved), Tensor.like(moved).fill(0))
    )

  /** A batch of records laid out in order, uploaded in four tensors rather than four per drawing. */
  def of[S: Label, Node: Label](records: Seq[RecordGraph], batch: Axis[S], nodes: AxisExtent[Node]): RecordBatch[S, Node] =
    val placed = records.map(_.placed(nodes.size))
    RecordBatch(
      nodeClass = Tensor2(batch, nodes.axis, VType[Int32]).fromArray(placed.map(_.nodeClass).toArray),
      xs = Tensor3(batch, nodes.axis, Axis[NodePoint], VType[Float32]).fromArray(placed.map(_.xs).toArray),
      ys = Tensor3(batch, nodes.axis, Axis[NodePoint], VType[Float32]).fromArray(placed.map(_.ys).toArray),
      links = Tensor3(batch, nodes.axis, Axis[NodeLink], VType[Int32]).fromArray(placed.map(_.links).toArray)
    )

/** One drawn node of a record. */
final case class RecordNode(nodeClass: NodeClass, points: Seq[Point])

/** One relationship of a record, by the index of the [[RecordNode]]s it links. */
final case class RecordEdge(edgeClass: NodeClass, subject: Int, obj: Int)

/** A record with nothing in it in any particular order, which is what a record is.
  *
  * Laying it out in order gives it one, reading it back takes the order away again, and that is
  * what makes two records comparable however they were written down.
  */
final case class RecordGraph(nodes: Seq[RecordNode], edges: Seq[RecordEdge]):

  def size: Int = nodes.length + edges.length

  /** The record laid out along the node axis: the nodes in their order, then the relationships. */
  def record[Node: Label](nodes: AxisExtent[Node]): Record[Node] =
    val laidOut = placed(nodes.size)
    Record(
      nodeClass = Tensor1(nodes.axis, VType[Int32]).fromArray(laidOut.nodeClass),
      xs = Tensor2(nodes.axis, Axis[NodePoint], VType[Float32]).fromArray(laidOut.xs),
      ys = Tensor2(nodes.axis, Axis[NodePoint], VType[Float32]).fromArray(laidOut.ys),
      links = Tensor2(nodes.axis, Axis[NodeLink], VType[Int32]).fromArray(laidOut.links)
    )

  /** The same record with its nodes and relationships in a random order. */
  def permuted(random: Random): RecordGraph =
    val order = random.shuffle(nodes.indices.toVector)
    val movedTo = order.zipWithIndex.toMap
    RecordGraph(
      nodes = order.map(nodes),
      edges = random.shuffle(edges).map(edge => RecordEdge(edge.edgeClass, movedTo(edge.subject), movedTo(edge.obj)))
    )

  private[dataset] def placed(slots: Int): RecordGraph.Placement =
    require(size <= slots, s"a record of $size nodes does not fit in $slots")
    val inOrder = nodes.map(node => (node.nodeClass, node.points, Seq(0, 0))) ++ edges.map: edge =>
      val ends = Seq(edge.subject, edge.obj)
      (edge.edgeClass, Seq.empty[Point], if edge.edgeClass.isSymmetric then ends.sorted else ends)
    RecordGraph.Placement(
      nodeClass = Array.tabulate(slots)(slot => inOrder.lift(slot).fold(NodeClass.NoNode)(_._1).id),
      xs = Array.tabulate(slots, NodeClass.maxPoints)((slot, at) => inOrder.lift(slot).flatMap(_._2.lift(at)).fold(0f)(_.x)),
      ys = Array.tabulate(slots, NodeClass.maxPoints)((slot, at) => inOrder.lift(slot).flatMap(_._2.lift(at)).fold(0f)(_.y)),
      links = Array.tabulate(slots, NodeClass.maxLinks)((slot, at) => inOrder.lift(slot).flatMap(_._3.lift(at)).getOrElse(0))
    )

object RecordGraph:

  /** A record laid out along the node axis, ready to be uploaded. */
  private[dataset] case class Placement(
      nodeClass: Array[Int],
      xs: Array[Array[Float]],
      ys: Array[Array[Float]],
      links: Array[Array[Int]]
  )

  /** The record a layout holds, with every relationship resolved to the nodes it links. One
    * naming a position that holds no drawn node is dropped: it relates nothing.
    */
  def of[Node: Label](record: Record[Node]): RecordGraph =
    read(record.nodeClass.toArray, record.xs.toArray, record.ys.toArray, record.links.toArray)

  /** Every record of a batch, read to the host — once for the batch, not once per drawing. */
  def of[S: Label, Node: Label](records: RecordBatch[S, Node]): Seq[RecordGraph] =
    val (nodeClass, xs, ys, links) = (records.nodeClass.toArray, records.xs.toArray, records.ys.toArray, records.links.toArray)
    nodeClass.indices.map(drawing => read(nodeClass(drawing), xs(drawing), ys(drawing), links(drawing)))

  private def read(nodeClasses: Array[Int], xs: Array[Array[Float]], ys: Array[Array[Float]], links: Array[Array[Int]]): RecordGraph =
    val nodeClass = nodeClasses.map(NodeClass.fromId)
    val nodeAt = nodeClass.indices.filter(nodeClass(_).isDrawn).zipWithIndex.toMap
    RecordGraph(
      nodes = nodeAt.keys.toSeq.sorted.map: at =>
        RecordNode(nodeClass(at), (0 until nodeClass(at).numPoints).map(point => Point(xs(at)(point), ys(at)(point)))),
      edges = nodeClass.indices.filter(nodeClass(_).isRelationship).flatMap: at =>
        for
          subject <- nodeAt.get(links(at)(0))
          obj <- nodeAt.get(links(at)(1))
        yield RecordEdge(nodeClass(at), subject, obj)
    )

  /** The nodes a detection holds, and nothing else — a detector predicts no relationships.
    *
    * A line is axis aligned, so the long side of its box is the line and the short side is what
    * [[Objects.of]] widened it to; an annotation is the centre of its box.
    */
  def of[Node: Label](detection: Detection[Node, Float32]): RecordGraph =
    val objectClass = detection.label.toArray.map(ObjectClass.fromId)
    val centerX = detection.box.centerX.toArray
    val centerY = detection.box.centerY.toArray
    val width = detection.box.width.toArray
    val height = detection.box.height.toArray
    RecordGraph(
      nodes = objectClass.indices.filter(objectClass(_) != ObjectClass.NoObject).map: at =>
        val (halfWidth, halfHeight) = (width(at) / 2, height(at) / 2)
        objectClass(at) match
          case ObjectClass.PartLine if width(at) >= height(at) =>
            RecordNode(NodeClass.Line, Seq(Point(centerX(at) - halfWidth, centerY(at)), Point(centerX(at) + halfWidth, centerY(at))))
          case ObjectClass.PartLine =>
            RecordNode(NodeClass.Line, Seq(Point(centerX(at), centerY(at) - halfHeight), Point(centerX(at), centerY(at) + halfHeight)))
          case _ =>
            RecordNode(NodeClass.Annotation, Seq(Point(centerX(at), centerY(at))))
      ,
      edges = Seq.empty
    )

  /** The record a drawing's objects hold. Relationships come out in the order a record lays them
    * out in, so that a record read back is the record it came from and not merely an isomorphic
    * one.
    */
  def of[Node: Label](objects: Objects[Node]): RecordGraph =
    val drawn = objects.detection.label.toArray.map(ObjectClass.fromId)
    val nodeAt = drawn.indices.filter(drawn(_) != ObjectClass.NoObject).zipWithIndex.toMap
    val relations = objects.relations.toArray
    of(objects.detection).copy(edges =
      for
        relation <- RelationClass.values.toSeq
        subject <- nodeAt.keys.toSeq.sorted
        obj <- nodeAt.keys.toSeq.sorted
        if relations(subject)(obj)(relation.id) > 0f
        if !relation.nodeClass.isSymmetric || subject < obj
      yield RecordEdge(relation.nodeClass, nodeAt(subject), nodeAt(obj))
    )
