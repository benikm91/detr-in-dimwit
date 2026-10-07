package dataset

import dimwit.*

import scala.util.Random

/** Axis over the [[NodeClass]] values a node of a record is classified into. */
trait NodeClasses derives Label

/** Axis over whether a node is construction geometry: 0 for part of the outline, 1 for a reference
  * the outline is constrained by, which a sketch draws dashed.
  */
trait IsConstruction derives Label

/** Axis over the [[EdgeClass]] values a relationship of a record is classified into. */
trait EdgeClasses derives Label

/** What a drawn node of a record is.
  *
  * [[NoNode]] marks a position the record does not reach, which is also where a transcription of
  * its nodes stops.
  */
enum NodeClass(val id: Int, val pointNames: Seq[String]):

  case NoNode extends NodeClass(0, Seq.empty)
  case Line extends NodeClass(1, Seq("start", "end"))
  case Annotation extends NodeClass(2, Seq("centre"))

  /** Placed by the two ends of its horizontal diameter, which give its centre and its radius. */
  case Circle extends NodeClass(3, Seq("left", "right"))

  /** Runs clockwise on the image from its start to its end, bending through the point halfway
    * along it.
    */
  case Arc extends NodeClass(4, Seq("start", "end", "mid"))

  /** A point of a sketch, drawn as a dot: a reference for constraints, such as a hole's centre. */
  case Point extends NodeClass(5, Seq("at"))

  def numPoints: Int = pointNames.length

  def isNode: Boolean = this != NodeClass.NoNode

object NodeClass:

  def fromId(id: Int): NodeClass =
    values.find(_.id == id).getOrElse(throw IllegalArgumentException(s"unknown node class id: $id"))

  def indicator[V: IsFloating](vtype: VType[V])(holds: NodeClass => Boolean): Tensor1[NodeClasses, V] =
    Tensor1(Axis[NodeClasses], vtype).fromArray(values.map(nodeClass => if holds(nodeClass) then 1f else 0f))

  /** The order the classes of a record are written down in, the simplest first: what a node needs
    * to be placed by grows from one point to three, and a node written later can be placed against
    * the ones before it. Within a class there is no order.
    */
  val WrittenInOrder: Seq[NodeClass] = Seq(Point, Line, Circle, Arc, Annotation)

  /** Where every class comes in [[WrittenInOrder]], and [[NoNode]] after all of them. */
  def writtenAt[V: IsFloating](vtype: VType[V]): Tensor1[NodeClasses, V] =
    Tensor1(Axis[NodeClasses], vtype).fromArray(values.map(nodeClass => WrittenInOrder.indexOf(nodeClass) match
      case -1 => WrittenInOrder.size.toFloat
      case at => at.toFloat
    ))

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

  def isEdge: Boolean = this != EdgeClass.NoEdge

  def numLinks: Int = if isEdge then 2 else 0

object EdgeClass:

  def fromId(id: Int): EdgeClass =
    values.find(_.id == id).getOrElse(throw IllegalArgumentException(s"unknown edge class id: $id"))

  def indicator[V: IsFloating](vtype: VType[V])(holds: EdgeClass => Boolean): Tensor1[EdgeClasses, V] =
    Tensor1(Axis[EdgeClasses], vtype).fromArray(values.map(edgeClass => if holds(edgeClass) then 1f else 0f))

/** A point of the canvas, normalized to it. */
case class Point(x: Float, y: Float)

/** The nodes a drawing draws, laid out along the `Node` axis.
  *
  * A node is placed by where it starts, if its class runs somewhere by where it ends, and if it
  * bends by its middle — a line by start and end, a circle by the ends of its diameter, an arc by
  * all three, an annotation by its start alone. What a class does not place is left at zero, and
  * the positions a record does not reach hold [[NodeClass.NoNode]] and are placed nowhere.
  * `construction` is one where a node is construction geometry, along [[IsConstruction]].
  */
case class RecordNodes[Node](
    nodeClass: Tensor1[Node, Int32],
    construction: Tensor1[Node, Int32],
    startX: Tensor1[Node, Float32],
    startY: Tensor1[Node, Float32],
    endX: Tensor1[Node, Float32],
    endY: Tensor1[Node, Float32],
    midX: Tensor1[Node, Float32],
    midY: Tensor1[Node, Float32]
)

/** The relationships between those nodes, laid out along the `Edge` axis.
  *
  * `subject` and `obj` name the nodes a relationship relates, by their position along the `Node`
  * axis. The positions a record does not reach hold [[EdgeClass.NoEdge]] and relate nothing.
  */
case class RecordEdges[Edge](
    edgeClass: Tensor1[Edge, Int32],
    subject: Tensor1[Edge, Int32],
    obj: Tensor1[Edge, Int32]
)

/** What a drawing encodes: the nodes it draws, and the relationships between them. */
case class Record[Node, Edge](nodes: RecordNodes[Node], edges: RecordEdges[Edge]):

  export nodes.{nodeClass, construction, startX, startY, endX, endY, midX, midY}
  export edges.{edgeClass, subject, obj}

object Record:

  def apply[Node, Edge](
      nodeClass: Tensor1[Node, Int32],
      construction: Tensor1[Node, Int32],
      startX: Tensor1[Node, Float32],
      startY: Tensor1[Node, Float32],
      endX: Tensor1[Node, Float32],
      endY: Tensor1[Node, Float32],
      midX: Tensor1[Node, Float32],
      midY: Tensor1[Node, Float32],
      edgeClass: Tensor1[Edge, Int32],
      subject: Tensor1[Edge, Int32],
      obj: Tensor1[Edge, Int32]
  ): Record[Node, Edge] =
    Record(RecordNodes(nodeClass, construction, startX, startY, endX, endY, midX, midY), RecordEdges(edgeClass, subject, obj))

/** [[Record]] for a batch of drawings along the axis `S`. */
case class RecordBatch[S, Node, Edge](
    nodeClass: Tensor2[S, Node, Int32],
    construction: Tensor2[S, Node, Int32],
    startX: Tensor2[S, Node, Float32],
    startY: Tensor2[S, Node, Float32],
    endX: Tensor2[S, Node, Float32],
    endY: Tensor2[S, Node, Float32],
    midX: Tensor2[S, Node, Float32],
    midY: Tensor2[S, Node, Float32],
    edgeClass: Tensor2[S, Edge, Int32],
    subject: Tensor2[S, Edge, Int32],
    obj: Tensor2[S, Edge, Int32]
):

  /** The same records, laid out again in `nodeSlots` and `edgeSlots` positions: their nodes class
    * by class in [[NodeClass.WrittenInOrder]], their relationships in any order, each in a fresh
    * random order within that, and the positions they do not reach last.
    *
    * What has no order is the model's to be indifferent to, which is what drawing a new one every
    * step is for. This one is drawn on the device — nothing is read back to lay it out again.
    */
  def permuted(key: Key, nodeSlots: AxisExtent[Node], edgeSlots: AxisExtent[Edge])(using Label[S], Label[Node], Label[Edge]): RecordBatch[S, Node, Edge] =
    val drawings = nodeClass.shape.extent(Axis[S])
    val padded = paddedTo(nodeSlots, edgeSlots, drawings)
    val (forNodes, forEdges) = key.split2()
    val (nodeKeys, edgeKeys) = (forNodes.splitToTensor(drawings), forEdges.splitToTensor(drawings))
    val (classes, constructions, startXs, startYs, endXs, endYs, midXs, midYs, nodeOrders) =
      zipvmap(Axis[S])(padded.nodeClass, padded.construction, padded.startX, padded.startY, padded.endX, padded.endY, padded.midX, padded.midY, nodeKeys):
        case (nodeClass, construction, startX, startY, endX, endY, midX, midY, key) =>
          RecordBatch.permutedNodes(RecordNodes(nodeClass, construction, startX, startY, endX, endY, midX, midY), key.item)
    val (edgeClasses, subjects, objs) =
      zipvmap(Axis[S])(padded.edgeClass, padded.subject, padded.obj, nodeOrders, edgeKeys):
        case (edgeClass, subject, obj, nodeOrder, key) =>
          RecordBatch.permutedEdges(RecordEdges(edgeClass, subject, obj), nodeOrder, key.item)
    RecordBatch(classes, constructions, startXs, startYs, endXs, endYs, midXs, midYs, edgeClasses, subjects, objs)

  /** The same records in as many positions, the ones they do not reach holding nothing. */
  private def paddedTo(nodeSlots: AxisExtent[Node], edgeSlots: AxisExtent[Edge], drawings: AxisExtent[S])(using Label[S], Label[Node], Label[Edge]): RecordBatch[S, Node, Edge] =
    val (heldNodes, heldEdges) = (nodeClass.shape(Axis[Node]), edgeClass.shape(Axis[Edge]))
    require(heldNodes <= nodeSlots.size, s"records of $heldNodes nodes do not fit in ${nodeSlots.size}")
    require(heldEdges <= edgeSlots.size, s"records of $heldEdges relationships do not fit in ${edgeSlots.size}")
    val (emptyNodes, emptyEdges) = (Axis[Node] -> (nodeSlots.size - heldNodes), Axis[Edge] -> (edgeSlots.size - heldEdges))
    def nowhere(placed: Tensor2[S, Node, Float32]) =
      concatenate(placed, Tensor(Shape(drawings, emptyNodes), VType[Float32]).fill(0f), Axis[Node])
    def nothing(named: Tensor2[S, Edge, Int32]) =
      concatenate(named, Tensor(Shape(drawings, emptyEdges), VType[Int32]).fill(0), Axis[Edge])
    RecordBatch(
      nodeClass = concatenate(nodeClass, Tensor(Shape(drawings, emptyNodes), VType[Int32]).fill(NodeClass.NoNode.id), Axis[Node]),
      construction = concatenate(construction, Tensor(Shape(drawings, emptyNodes), VType[Int32]).fill(0), Axis[Node]),
      startX = nowhere(startX),
      startY = nowhere(startY),
      endX = nowhere(endX),
      endY = nowhere(endY),
      midX = nowhere(midX),
      midY = nowhere(midY),
      edgeClass = concatenate(edgeClass, Tensor(Shape(drawings, emptyEdges), VType[Int32]).fill(EdgeClass.NoEdge.id), Axis[Edge]),
      subject = nothing(subject),
      obj = nothing(obj)
    )

object RecordBatch:

  /** One record's nodes class by class, in a fresh random order within a class, with the
    * positions it does not reach last, and the order they were read in — which is what its
    * relationships name them by.
    */
  private def permutedNodes[Node: Label](nodes: RecordNodes[Node], key: Key): (
      Tensor1[Node, Int32],
      Tensor1[Node, Int32],
      Tensor1[Node, Float32],
      Tensor1[Node, Float32],
      Tensor1[Node, Float32],
      Tensor1[Node, Float32],
      Tensor1[Node, Float32],
      Tensor1[Node, Float32],
      Tensor1[Node, Int32]
  ) =
    val order = inOrder(NodeClass.writtenAt(VType[Float32]).slice(Axis[NodeClasses].at(nodes.nodeClass)), key)
    def reordered[V](placed: Tensor1[Node, V]) = placed.slice(Axis[Node].at(order))
    (
      reordered(nodes.nodeClass),
      reordered(nodes.construction),
      reordered(nodes.startX),
      reordered(nodes.startY),
      reordered(nodes.endX),
      reordered(nodes.endY),
      reordered(nodes.midX),
      reordered(nodes.midY),
      order
    )

  /** One record's relationships in a fresh random order, naming the nodes by where `nodeOrder`
    * has just put them.
    */
  private def permutedEdges[Node: Label, Edge: Label](
      edges: RecordEdges[Edge],
      nodeOrder: Tensor1[Node, Int32],
      key: Key
  ): (Tensor1[Edge, Int32], Tensor1[Edge, Int32], Tensor1[Edge, Int32]) =
    val order = inOrder(EdgeClass.indicator(VType[Float32])(!_.isEdge).slice(Axis[EdgeClasses].at(edges.edgeClass)), key)
    val classes = edges.edgeClass.slice(Axis[Edge].at(order))
    // A relationship names the nodes it relates by their position, and a node that sat at `at`
    // before sits at `renamed(at)` now.
    val renamed = nodeOrder.argsort(Axis[Node])
    def moved(named: Tensor1[Edge, Int32]) = renamed.slice(Axis[Node].at(named.slice(Axis[Edge].at(order))))
    val (subject, obj) = (moved(edges.subject), moved(edges.obj))
    def is(holds: EdgeClass => Boolean) =
      val marked = EdgeClass.indicator(VType[Float32])(holds).slice(Axis[EdgeClasses].at(classes))
      marked > Tensor.like(marked).fill(0f)
    // A symmetric relationship names the two it relates in ascending order, and a position
    // holding no relationship relates nothing.
    val nothing = Tensor.like(subject).fill(0)
    (
      classes,
      where(is(_.isEdge), where(is(_.isSymmetric), minimum(subject, obj), subject), nothing),
      where(is(_.isEdge), where(is(_.isSymmetric), maximum(subject, obj), obj), nothing)
    )

  /** The order that reads the positions by ascending `rank`, shuffled within a rank. */
  private def inOrder[L: Label](rank: Tensor1[L, Float32], key: Key): Tensor1[L, Int32] =
    val slots = rank.shape.extent(Axis[L])
    val shuffle = dimwit.Random.permutation(slots)(key).asFloat(VType[Float32])
    (rank *! slots.size.toFloat + shuffle).argsort(Axis[L])

  /** A batch of records laid out in order, uploaded in five tensors rather than five per drawing. */
  def of[S: Label, Node: Label, Edge: Label](records: Seq[RecordGraph], batch: Axis[S], nodes: AxisExtent[Node], edges: AxisExtent[Edge]): RecordBatch[S, Node, Edge] =
    val placed = records.map(_.placed(nodes.size, edges.size))
    RecordBatch(
      nodeClass = Tensor2(batch, nodes.axis, VType[Int32]).fromArray(placed.map(_.nodeClass).toArray),
      construction = Tensor2(batch, nodes.axis, VType[Int32]).fromArray(placed.map(_.construction).toArray),
      startX = Tensor2(batch, nodes.axis, VType[Float32]).fromArray(placed.map(_.startX).toArray),
      startY = Tensor2(batch, nodes.axis, VType[Float32]).fromArray(placed.map(_.startY).toArray),
      endX = Tensor2(batch, nodes.axis, VType[Float32]).fromArray(placed.map(_.endX).toArray),
      endY = Tensor2(batch, nodes.axis, VType[Float32]).fromArray(placed.map(_.endY).toArray),
      midX = Tensor2(batch, nodes.axis, VType[Float32]).fromArray(placed.map(_.midX).toArray),
      midY = Tensor2(batch, nodes.axis, VType[Float32]).fromArray(placed.map(_.midY).toArray),
      edgeClass = Tensor2(batch, edges.axis, VType[Int32]).fromArray(placed.map(_.edgeClass).toArray),
      subject = Tensor2(batch, edges.axis, VType[Int32]).fromArray(placed.map(_.subject).toArray),
      obj = Tensor2(batch, edges.axis, VType[Int32]).fromArray(placed.map(_.obj).toArray)
    )

/** One drawn node of a record. */
case class RecordNode(nodeClass: NodeClass, isConstruction: Boolean, points: Seq[Point])

/** One relationship of a record, by the index of the [[RecordNode]]s it links. */
case class RecordEdge(edgeClass: EdgeClass, subject: Int, obj: Int)

/** A record with nothing in it in any particular order, which is what a record is.
  *
  * Laying it out in order gives it one, reading it back takes the order away again, and that is
  * what makes two records comparable however they were written down.
  */
case class RecordGraph(nodes: Seq[RecordNode], edges: Seq[RecordEdge]):

  def size: Int = nodes.length + edges.length

  /** The record laid out along the node and the edge axis, each in its own order. */
  def record[Node: Label, Edge: Label](nodes: AxisExtent[Node], edges: AxisExtent[Edge]): Record[Node, Edge] =
    val laidOut = placed(nodes.size, edges.size)
    def placedAt(read: RecordGraph.Placement => Array[Float]) = Tensor1(nodes.axis, VType[Float32]).fromArray(read(laidOut))
    def named(read: RecordGraph.Placement => Array[Int]) = Tensor1(edges.axis, VType[Int32]).fromArray(read(laidOut))
    Record(
      nodeClass = Tensor1(nodes.axis, VType[Int32]).fromArray(laidOut.nodeClass),
      construction = Tensor1(nodes.axis, VType[Int32]).fromArray(laidOut.construction),
      startX = placedAt(_.startX),
      startY = placedAt(_.startY),
      endX = placedAt(_.endX),
      endY = placedAt(_.endY),
      midX = placedAt(_.midX),
      midY = placedAt(_.midY),
      edgeClass = named(_.edgeClass),
      subject = named(_.subject),
      obj = named(_.obj)
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
    def placedAt(of: Point => Float, at: Int) =
      Array.tabulate(nodeSlots)(slot => nodes.lift(slot).flatMap(_.points.lift(at)).fold(0f)(of))
    def named(end: Int) = Array.tabulate(edgeSlots)(slot => related.lift(slot).flatMap(_._2.lift(end)).getOrElse(0))
    RecordGraph.Placement(
      nodeClass = Array.tabulate(nodeSlots)(slot => nodes.lift(slot).fold(NodeClass.NoNode)(_.nodeClass).id),
      construction = Array.tabulate(nodeSlots)(slot => if nodes.lift(slot).exists(_.isConstruction) then 1 else 0),
      startX = placedAt(_.x, 0),
      startY = placedAt(_.y, 0),
      endX = placedAt(_.x, 1),
      endY = placedAt(_.y, 1),
      midX = placedAt(_.x, 2),
      midY = placedAt(_.y, 2),
      edgeClass = Array.tabulate(edgeSlots)(slot => related.lift(slot).fold(EdgeClass.NoEdge)(_._1).id),
      subject = named(0),
      obj = named(1)
    )

object RecordGraph:

  /** A record laid out along its two axes, ready to be uploaded. */
  private[dataset] case class Placement(
      nodeClass: Array[Int],
      construction: Array[Int],
      startX: Array[Float],
      startY: Array[Float],
      endX: Array[Float],
      endY: Array[Float],
      midX: Array[Float],
      midY: Array[Float],
      edgeClass: Array[Int],
      subject: Array[Int],
      obj: Array[Int]
  )

  /** The record a layout holds, with every relationship resolved to the nodes it links. One
    * naming a position that holds no drawn node is dropped: it relates nothing.
    */
  def of[Node: Label, Edge: Label](record: Record[Node, Edge]): RecordGraph =
    read(
      RecordGraph.Placement(
        record.nodeClass.toArray,
        record.construction.toArray,
        record.startX.toArray,
        record.startY.toArray,
        record.endX.toArray,
        record.endY.toArray,
        record.midX.toArray,
        record.midY.toArray,
        record.edgeClass.toArray,
        record.subject.toArray,
        record.obj.toArray
      )
    )

  /** The nodes alone, for a model that predicts no relationships. */
  def of[Node: Label](nodes: RecordNodes[Node]): RecordGraph =
    read(
      RecordGraph.Placement(
        nodes.nodeClass.toArray,
        nodes.construction.toArray,
        nodes.startX.toArray,
        nodes.startY.toArray,
        nodes.endX.toArray,
        nodes.endY.toArray,
        nodes.midX.toArray,
        nodes.midY.toArray,
        Array.empty,
        Array.empty,
        Array.empty
      )
    )

  /** Every record of a batch, read to the host — once for the batch, not once per drawing. */
  def of[S: Label, Node: Label, Edge: Label](records: RecordBatch[S, Node, Edge]): Seq[RecordGraph] =
    val (nodeClass, construction) = (records.nodeClass.toArray, records.construction.toArray)
    val (startX, startY) = (records.startX.toArray, records.startY.toArray)
    val (endX, endY) = (records.endX.toArray, records.endY.toArray)
    val (midX, midY) = (records.midX.toArray, records.midY.toArray)
    // DimWit's `toArray` returns no rows at all for a tensor whose second axis is empty.
    def perDrawing(named: Tensor2[S, Edge, Int32]) =
      if named.shape(Axis[Edge]) == 0 then Array.fill(nodeClass.length)(Array.empty[Int]) else named.toArray
    val (edgeClass, subject, obj) = (perDrawing(records.edgeClass), perDrawing(records.subject), perDrawing(records.obj))
    nodeClass.indices.map: drawing =>
      read(
        RecordGraph.Placement(
          nodeClass(drawing),
          construction(drawing),
          startX(drawing),
          startY(drawing),
          endX(drawing),
          endY(drawing),
          midX(drawing),
          midY(drawing),
          edgeClass(drawing),
          subject(drawing),
          obj(drawing)
        )
      )

  private def read(placed: Placement): RecordGraph =
    val nodeClass = placed.nodeClass.map(NodeClass.fromId)
    val edgeClass = placed.edgeClass.map(EdgeClass.fromId)
    val nodeAt = nodeClass.indices.filter(nodeClass(_).isNode).zipWithIndex.toMap
    def placedAt(at: Int) =
      Seq(Point(placed.startX(at), placed.startY(at)), Point(placed.endX(at), placed.endY(at)), Point(placed.midX(at), placed.midY(at)))
    RecordGraph(
      nodes = nodeAt.keys.toSeq.sorted.map(at => RecordNode(nodeClass(at), placed.construction(at) == 1, placedAt(at).take(nodeClass(at).numPoints))),
      edges = edgeClass.indices.filter(edgeClass(_).isEdge).flatMap: at =>
        for
          subject <- nodeAt.get(placed.subject(at))
          obj <- nodeAt.get(placed.obj(at))
        yield RecordEdge(edgeClass(at), subject, obj)
    )
