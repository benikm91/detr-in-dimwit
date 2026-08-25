package dataset

import dimwit.*

import scala.util.Random

/** Axis over the [[NodeClass]] values a node of a record is classified into. */
trait NodeClasses derives Label

/** Axis over the [[EdgeClass]] values a relationship of a record is classified into. */
trait EdgeClasses derives Label

/** Axis of the points a node is placed by: a line has two, an annotation one. */
trait NodePoint derives Label

/** Axis of the nodes a relationship links: its subject, then its object. */
trait NodeLink derives Label

/** What a drawn node of a record is.
  *
  * [[NoNode]] marks a position the record does not reach, which is also where a transcription of
  * its nodes stops.
  */
enum NodeClass(val id: Int, val pointNames: Seq[String]):

  case NoNode extends NodeClass(0, Seq.empty)
  case Line extends NodeClass(1, Seq("start", "end"))
  case Annotation extends NodeClass(2, Seq("centre"))

  def numPoints: Int = pointNames.length

  def isDrawn: Boolean = this != NodeClass.NoNode

object NodeClass:

  def fromId(id: Int): NodeClass =
    values.find(_.id == id).getOrElse(throw IllegalArgumentException(s"unknown node class id: $id"))

  val maxPoints: Int = values.map(_.numPoints).max

  def indicator[V: IsFloating](vtype: VType[V])(holds: NodeClass => Boolean): Tensor1[NodeClasses, V] =
    Tensor1(Axis[NodeClasses], vtype).fromArray(values.map(nodeClass => if holds(nodeClass) then 1f else 0f))

  /** Which points a class places itself by, so that a node is only measured on what it carries. */
  def usedPoints[V: IsFloating](vtype: VType[V]): Tensor2[NodeClasses, NodePoint, V] =
    Tensor2(Axis[NodeClasses], Axis[NodePoint], vtype).fromArray(
      values.map(nodeClass => Array.tabulate(maxPoints)(point => if point < nodeClass.numPoints then 1f else 0f))
    )

/** What a relationship of a record is, which a record holds as a node of its own so that a graph
  * is a set.
  *
  * [[NoEdge]] marks a position the record does not reach, which is also where a transcription of
  * its relationships stops.
  */
enum EdgeClass(val id: Int, val isSymmetric: Boolean):

  case NoEdge extends EdgeClass(0, false)

  /** Two lines meeting in a corner, held once with the two it links in ascending order. */
  case Connected extends EdgeClass(1, true)

  case Annotates extends EdgeClass(2, false)

  def relates: Boolean = this != EdgeClass.NoEdge

  def numLinks: Int = if relates then 2 else 0

object EdgeClass:

  def fromId(id: Int): EdgeClass =
    values.find(_.id == id).getOrElse(throw IllegalArgumentException(s"unknown edge class id: $id"))

  val maxLinks: Int = values.map(_.numLinks).max

  def indicator[V: IsFloating](vtype: VType[V])(holds: EdgeClass => Boolean): Tensor1[EdgeClasses, V] =
    Tensor1(Axis[EdgeClasses], vtype).fromArray(values.map(edgeClass => if holds(edgeClass) then 1f else 0f))

  /** Which nodes a class links, which is both of them for a relationship and neither otherwise. */
  def usedLinks[V: IsFloating](vtype: VType[V]): Tensor2[EdgeClasses, NodeLink, V] =
    Tensor2(Axis[EdgeClasses], Axis[NodeLink], vtype).fromArray(
      values.map(edgeClass => Array.tabulate(maxLinks)(link => if link < edgeClass.numLinks then 1f else 0f))
    )

/** A point of the canvas, normalized to it. */
final case class Point(x: Float, y: Float)

/** The nodes a drawing draws, laid out along the `Node` axis.
  *
  * `xs` and `ys` place a node on the canvas, as many points as its class carries. The positions a
  * record does not reach hold [[NodeClass.NoNode]].
  */
final case class RecordNodes[Node](
    nodeClass: Tensor1[Node, Int32],
    xs: Tensor2[Node, NodePoint, Float32],
    ys: Tensor2[Node, NodePoint, Float32]
)

/** The relationships between those nodes, laid out along the `Edge` axis.
  *
  * `links` name the nodes a relationship relates, by their position along the `Node` axis. The
  * positions a record does not reach hold [[EdgeClass.NoEdge]] and link nothing.
  */
final case class RecordEdges[Edge](
    edgeClass: Tensor1[Edge, Int32],
    links: Tensor2[Edge, NodeLink, Int32]
)

/** What a drawing encodes: the nodes it draws, and the relationships between them. */
final case class Record[Node, Edge](nodes: RecordNodes[Node], edges: RecordEdges[Edge]):

  export nodes.{nodeClass, xs, ys}
  export edges.{edgeClass, links}

object Record:

  def apply[Node, Edge](
      nodeClass: Tensor1[Node, Int32],
      xs: Tensor2[Node, NodePoint, Float32],
      ys: Tensor2[Node, NodePoint, Float32],
      edgeClass: Tensor1[Edge, Int32],
      links: Tensor2[Edge, NodeLink, Int32]
  ): Record[Node, Edge] = Record(RecordNodes(nodeClass, xs, ys), RecordEdges(edgeClass, links))

/** [[Record]] for a batch of drawings along the axis `S`. */
final case class RecordBatch[S, Node, Edge](
    nodeClass: Tensor2[S, Node, Int32],
    xs: Tensor3[S, Node, NodePoint, Float32],
    ys: Tensor3[S, Node, NodePoint, Float32],
    edgeClass: Tensor2[S, Edge, Int32],
    links: Tensor3[S, Edge, NodeLink, Int32]
):

  /** The same records, laid out again in `nodeSlots` and `edgeSlots` positions with their nodes
    * and their relationships each in a fresh random order, and the positions they do not reach
    * last.
    *
    * A record is a set, so the order it is written down in is the model's to be indifferent to,
    * which is what drawing a new one every step is for. This one is drawn on the device — nothing
    * is read back to lay it out again.
    */
  def permuted(key: Key, nodeSlots: AxisExtent[Node], edgeSlots: AxisExtent[Edge])(using Label[S], Label[Node], Label[Edge]): RecordBatch[S, Node, Edge] =
    val drawings = nodeClass.shape.extent(Axis[S])
    val padded = paddedTo(nodeSlots, edgeSlots, drawings)
    val (forNodes, forEdges) = key.split2()
    val (nodeKeys, edgeKeys) = (forNodes.splitToTensor(drawings), forEdges.splitToTensor(drawings))
    val (classes, permutedXs, permutedYs, nodeOrders) =
      zipvmap(Axis[S])(padded.nodeClass, padded.xs, padded.ys, nodeKeys):
        case (nodeClass, xs, ys, key) => RecordBatch.permutedNodes(nodeClass, xs, ys, key.item)
    val (edgeClasses, permutedLinks) =
      zipvmap(Axis[S])(padded.edgeClass, padded.links, nodeOrders, edgeKeys):
        case (edgeClass, links, nodeOrder, key) => RecordBatch.permutedEdges(edgeClass, links, nodeOrder, key.item)
    RecordBatch(classes, permutedXs, permutedYs, edgeClasses, permutedLinks)

  /** The same records in as many positions, the ones they do not reach holding nothing. */
  private def paddedTo(nodeSlots: AxisExtent[Node], edgeSlots: AxisExtent[Edge], drawings: AxisExtent[S])(using Label[S], Label[Node], Label[Edge]): RecordBatch[S, Node, Edge] =
    val (heldNodes, heldEdges) = (nodeClass.shape(Axis[Node]), edgeClass.shape(Axis[Edge]))
    require(heldNodes <= nodeSlots.size, s"records of $heldNodes nodes do not fit in ${nodeSlots.size}")
    require(heldEdges <= edgeSlots.size, s"records of $heldEdges relationships do not fit in ${edgeSlots.size}")
    val (emptyNodes, emptyEdges) = (Axis[Node] -> (nodeSlots.size - heldNodes), Axis[Edge] -> (edgeSlots.size - heldEdges))
    val points = Axis[NodePoint] -> NodeClass.maxPoints
    RecordBatch(
      nodeClass = concatenate(nodeClass, Tensor(Shape(drawings, emptyNodes), VType[Int32]).fill(NodeClass.NoNode.id), Axis[Node]),
      xs = concatenate(xs, Tensor(Shape(drawings, emptyNodes, points), VType[Float32]).fill(0f), Axis[Node]),
      ys = concatenate(ys, Tensor(Shape(drawings, emptyNodes, points), VType[Float32]).fill(0f), Axis[Node]),
      edgeClass = concatenate(edgeClass, Tensor(Shape(drawings, emptyEdges), VType[Int32]).fill(EdgeClass.NoEdge.id), Axis[Edge]),
      links = concatenate(links, Tensor(Shape(drawings, emptyEdges, Axis[NodeLink] -> EdgeClass.maxLinks), VType[Int32]).fill(0), Axis[Edge])
    )

object RecordBatch:

  /** One record's nodes in a fresh random order, with the positions it does not reach last, and
    * the order they were read in — which is what its relationships name them by.
    */
  private def permutedNodes[Node: Label](
      nodeClass: Tensor1[Node, Int32],
      xs: Tensor2[Node, NodePoint, Float32],
      ys: Tensor2[Node, NodePoint, Float32],
      key: Key
  ): (Tensor1[Node, Int32], Tensor2[Node, NodePoint, Float32], Tensor2[Node, NodePoint, Float32], Tensor1[Node, Int32]) =
    val order = heldFirst(NodeClass.indicator(VType[Float32])(_.isDrawn).take(Axis[NodeClasses])(nodeClass), key)
    (nodeClass.take(Axis[Node])(order), xs.take(Axis[Node])(order), ys.take(Axis[Node])(order), order)

  /** One record's relationships in a fresh random order, naming the nodes by where `nodeOrder`
    * has just put them.
    */
  private def permutedEdges[Node: Label, Edge: Label](
      edgeClass: Tensor1[Edge, Int32],
      links: Tensor2[Edge, NodeLink, Int32],
      nodeOrder: Tensor1[Node, Int32],
      key: Key
  ): (Tensor1[Edge, Int32], Tensor2[Edge, NodeLink, Int32]) =
    val order = heldFirst(EdgeClass.indicator(VType[Float32])(_.relates).take(Axis[EdgeClasses])(edgeClass), key)
    val classes = edgeClass.take(Axis[Edge])(order)
    // A relationship names the nodes it links by their position, and a node that sat at `at`
    // before sits at `renamed(at)` now.
    val renamed = nodeOrder.argsort(Axis[Node])
    val moved = links.take(Axis[Edge])(order).vmap(Axis[NodeLink])(renamed.take(Axis[Node])(_))
    val ascending = stack(Seq(moved.min(Axis[NodeLink]), moved.max(Axis[NodeLink])), newAxis = Axis[NodeLink], afterAxis = Axis[Edge])
    def is(holds: EdgeClass => Boolean) =
      val marked = EdgeClass.indicator(VType[Float32])(holds).take(Axis[EdgeClasses])(classes)
      marked > Tensor.like(marked).fill(0f)
    // A symmetric relationship names the two it links in ascending order, and a position holding
    // no relationship names position zero.
    (classes, where_!(is(_.relates), where_!(is(_.isSymmetric), ascending, moved), Tensor.like(moved).fill(0)))

  /** The order that reads the positions `holds` marks first, shuffled, and the empty ones after
    * them.
    */
  private def heldFirst[L: Label](holds: Tensor1[L, Float32], key: Key): Tensor1[L, Int32] =
    val slots = holds.shape.extent(Axis[L])
    val shuffle = dimwit.Random.permutation(slots)(key).asFloat(VType[Float32])
    ((Tensor.like(holds).fill(1f) - holds) * Tensor.like(holds).fill(slots.size.toFloat) + shuffle).argsort(Axis[L])

  /** A batch of records laid out in order, uploaded in five tensors rather than five per drawing. */
  def of[S: Label, Node: Label, Edge: Label](records: Seq[RecordGraph], batch: Axis[S], nodes: AxisExtent[Node], edges: AxisExtent[Edge]): RecordBatch[S, Node, Edge] =
    val placed = records.map(_.placed(nodes.size, edges.size))
    RecordBatch(
      nodeClass = Tensor2(batch, nodes.axis, VType[Int32]).fromArray(placed.map(_.nodeClass).toArray),
      xs = Tensor3(batch, nodes.axis, Axis[NodePoint], VType[Float32]).fromArray(placed.map(_.xs).toArray),
      ys = Tensor3(batch, nodes.axis, Axis[NodePoint], VType[Float32]).fromArray(placed.map(_.ys).toArray),
      edgeClass = Tensor2(batch, edges.axis, VType[Int32]).fromArray(placed.map(_.edgeClass).toArray),
      links = Tensor3(batch, edges.axis, Axis[NodeLink], VType[Int32]).fromArray(placed.map(_.links).toArray)
    )

/** One drawn node of a record. */
final case class RecordNode(nodeClass: NodeClass, points: Seq[Point])

/** One relationship of a record, by the index of the [[RecordNode]]s it links. */
final case class RecordEdge(edgeClass: EdgeClass, subject: Int, obj: Int)

/** A record with nothing in it in any particular order, which is what a record is.
  *
  * Laying it out in order gives it one, reading it back takes the order away again, and that is
  * what makes two records comparable however they were written down.
  */
final case class RecordGraph(nodes: Seq[RecordNode], edges: Seq[RecordEdge]):

  def size: Int = nodes.length + edges.length

  /** The record laid out along the node and the edge axis, each in its own order. */
  def record[Node: Label, Edge: Label](nodes: AxisExtent[Node], edges: AxisExtent[Edge]): Record[Node, Edge] =
    val laidOut = placed(nodes.size, edges.size)
    Record(
      nodeClass = Tensor1(nodes.axis, VType[Int32]).fromArray(laidOut.nodeClass),
      xs = Tensor2(nodes.axis, Axis[NodePoint], VType[Float32]).fromArray(laidOut.xs),
      ys = Tensor2(nodes.axis, Axis[NodePoint], VType[Float32]).fromArray(laidOut.ys),
      edgeClass = Tensor1(edges.axis, VType[Int32]).fromArray(laidOut.edgeClass),
      links = Tensor2(edges.axis, Axis[NodeLink], VType[Int32]).fromArray(laidOut.links)
    )

  /** The same record with its nodes and its relationships in a random order. */
  def permuted(random: Random): RecordGraph =
    val order = random.shuffle(nodes.indices.toVector)
    val movedTo = order.zipWithIndex.toMap
    RecordGraph(
      nodes = order.map(nodes),
      edges = random.shuffle(edges).map(edge => RecordEdge(edge.edgeClass, movedTo(edge.subject), movedTo(edge.obj)))
    )

  private[dataset] def placed(nodeSlots: Int, edgeSlots: Int): RecordGraph.Placement =
    require(nodes.size <= nodeSlots, s"a record of ${nodes.size} nodes does not fit in $nodeSlots")
    require(edges.size <= edgeSlots, s"a record of ${edges.size} relationships does not fit in $edgeSlots")
    val related = edges.map: edge =>
      val ends = Seq(edge.subject, edge.obj)
      (edge.edgeClass, if edge.edgeClass.isSymmetric then ends.sorted else ends)
    RecordGraph.Placement(
      nodeClass = Array.tabulate(nodeSlots)(slot => nodes.lift(slot).fold(NodeClass.NoNode)(_.nodeClass).id),
      xs = Array.tabulate(nodeSlots, NodeClass.maxPoints)((slot, at) => nodes.lift(slot).flatMap(_.points.lift(at)).fold(0f)(_.x)),
      ys = Array.tabulate(nodeSlots, NodeClass.maxPoints)((slot, at) => nodes.lift(slot).flatMap(_.points.lift(at)).fold(0f)(_.y)),
      edgeClass = Array.tabulate(edgeSlots)(slot => related.lift(slot).fold(EdgeClass.NoEdge)(_._1).id),
      links = Array.tabulate(edgeSlots, EdgeClass.maxLinks)((slot, at) => related.lift(slot).flatMap(_._2.lift(at)).getOrElse(0))
    )

object RecordGraph:

  /** A record laid out along its two axes, ready to be uploaded. */
  private[dataset] case class Placement(
      nodeClass: Array[Int],
      xs: Array[Array[Float]],
      ys: Array[Array[Float]],
      edgeClass: Array[Int],
      links: Array[Array[Int]]
  )

  /** The record a layout holds, with every relationship resolved to the nodes it links. One
    * naming a position that holds no drawn node is dropped: it relates nothing.
    */
  def of[Node: Label, Edge: Label](record: Record[Node, Edge]): RecordGraph =
    read(record.nodeClass.toArray, record.xs.toArray, record.ys.toArray, record.edgeClass.toArray, record.links.toArray)

  /** Every record of a batch, read to the host — once for the batch, not once per drawing. */
  def of[S: Label, Node: Label, Edge: Label](records: RecordBatch[S, Node, Edge]): Seq[RecordGraph] =
    val (nodeClass, xs, ys) = (records.nodeClass.toArray, records.xs.toArray, records.ys.toArray)
    val (edgeClass, links) = (records.edgeClass.toArray, records.links.toArray)
    nodeClass.indices.map(drawing => read(nodeClass(drawing), xs(drawing), ys(drawing), edgeClass(drawing), links(drawing)))

  private def read(
      nodeClasses: Array[Int],
      xs: Array[Array[Float]],
      ys: Array[Array[Float]],
      edgeClasses: Array[Int],
      links: Array[Array[Int]]
  ): RecordGraph =
    val nodeClass = nodeClasses.map(NodeClass.fromId)
    val edgeClass = edgeClasses.map(EdgeClass.fromId)
    val nodeAt = nodeClass.indices.filter(nodeClass(_).isDrawn).zipWithIndex.toMap
    RecordGraph(
      nodes = nodeAt.keys.toSeq.sorted.map: at =>
        RecordNode(nodeClass(at), (0 until nodeClass(at).numPoints).map(point => Point(xs(at)(point), ys(at)(point)))),
      edges = edgeClass.indices.filter(edgeClass(_).relates).flatMap: at =>
        for
          subject <- nodeAt.get(links(at)(0))
          obj <- nodeAt.get(links(at)(1))
        yield RecordEdge(edgeClass(at), subject, obj)
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
        if !relation.edgeClass.isSymmetric || subject < obj
      yield RecordEdge(relation.edgeClass, nodeAt(subject), nodeAt(obj))
    )
