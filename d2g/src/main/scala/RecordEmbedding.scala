import dataset.EdgeClass
import dataset.EdgeClasses
import dataset.NodeClass
import dataset.NodeClasses
import dataset.RecordEdges
import dataset.RecordNodes
import deepwit.base.AffineLayer
import dimwit.*

/** Reads a record's nodes into one embedding each: the class and the points it is placed by
  * embedded on their own, concatenated and projected into the space the decoder works in.
  */
class NodeEmbedder[V: IsFloating](params: NodeEmbedder.Params[V], canvas: Int)
    extends (RecordNodes[Node] => Tensor2[Node, Embedding, V]):

  private val project = AffineLayer(params.projection)

  override def apply(nodes: RecordNodes[Node]): Tensor2[Node, Embedding, V] =
    def placed(table: Tensor2[Pixel, PartEmbedding, V], coordinate: Tensor1[Node, Float32]) =
      table.take(Axis[Pixel])(Pixels.of(coordinate, canvas))
    val parts = Seq(
      params.nodeClass.take(Axis[NodeClasses])(nodes.nodeClass),
      placed(params.startX, nodes.startX),
      placed(params.startY, nodes.startY),
      placed(params.endX, nodes.endX),
      placed(params.endY, nodes.endY)
    )
    stack(parts, Axis[NodePart])
      .swap(Axis[NodePart], Axis[Node])
      .flatten((Axis[NodePart], Axis[PartEmbedding]))
      .vmap(Axis[Node])(project)

object NodeEmbedder:

  case class Params[V](
      nodeClass: Tensor2[NodeClasses, PartEmbedding, V],
      startX: Tensor2[Pixel, PartEmbedding, V],
      startY: Tensor2[Pixel, PartEmbedding, V],
      endX: Tensor2[Pixel, PartEmbedding, V],
      endY: Tensor2[Pixel, PartEmbedding, V],
      projection: AffineLayer.Params[NodePart |*| PartEmbedding, Embedding, V]
  )

  object Params:
    given tensorTree: TensorTree[Params[Float32]] = TensorTree.derived
    given tree: TreeOf[Params[Float32], Float32] = TreeOf.derived

/** Reads a record's relationships into one embedding each: the class and the nodes it relates
  * embedded on their own, concatenated and projected into the space the decoder works in.
  */
class EdgeEmbedder[V: IsFloating](params: EdgeEmbedder.Params[V])
    extends (RecordEdges[Edge] => Tensor2[Edge, Embedding, V]):

  private val project = AffineLayer(params.projection)

  override def apply(edges: RecordEdges[Edge]): Tensor2[Edge, Embedding, V] =
    def named(table: Tensor2[LinkedNode, PartEmbedding, V], end: Tensor1[Edge, Int32]) =
      table.take(Axis[LinkedNode])(end)
    val parts = Seq(
      params.edgeClass.take(Axis[EdgeClasses])(edges.edgeClass),
      named(params.subject, edges.subject),
      named(params.obj, edges.obj)
    )
    stack(parts, Axis[EdgePart])
      .swap(Axis[EdgePart], Axis[Edge])
      .flatten((Axis[EdgePart], Axis[PartEmbedding]))
      .vmap(Axis[Edge])(project)

object EdgeEmbedder:

  case class Params[V](
      edgeClass: Tensor2[EdgeClasses, PartEmbedding, V],
      subject: Tensor2[LinkedNode, PartEmbedding, V],
      obj: Tensor2[LinkedNode, PartEmbedding, V],
      projection: AffineLayer.Params[EdgePart |*| PartEmbedding, Embedding, V]
  )

  object Params:
    given tensorTree: TensorTree[Params[Float32]] = TensorTree.derived
    given tree: TreeOf[Params[Float32], Float32] = TreeOf.derived

/** Reads an embedding per position back into a record's nodes, and settles them. */
class NodeScorer[V: IsFloating](params: NodeScorer.Params[V])
    extends (Tensor2[Node, Embedding, V] => NodeScorer.NodeLogits[V]):

  import NodeScorer.NodeLogits

  private val classify = AffineLayer(params.nodeClass)

  /** How wide the canvas a coordinate is scored on is, i.e. how fine a pixel is. */
  val canvas: Int = params.startX.bias.shape(Axis[Pixel])

  override def apply(embeddings: Tensor2[Node, Embedding, V]): NodeLogits[V] =
    def placed(coordinate: AffineLayer.Params[Embedding, Pixel, V]) =
      embeddings.vmap(Axis[Node])(AffineLayer(coordinate))
    NodeLogits(
      nodeClass = embeddings.vmap(Axis[Node])(classify),
      startX = placed(params.startX),
      startY = placed(params.startY),
      endX = placed(params.endX),
      endY = placed(params.endY)
    )

  /** The scores settled. Settling on [[NodeClass.NoNode]] is where the nodes of a record stop.
    *
    * A node is placed only where its class places itself, so the rest is cleared rather than left
    * at whatever an unsupervised head happened to say — a node the model takes has to look like a
    * node the data would have given it.
    */
  def decide(logits: NodeLogits[V]): RecordNodes[Node] =
    val nodeClass = logits.nodeClass.argmax(Axis[NodeClasses])
    def carries(holds: NodeClass => Boolean) = NodeClass.indicator(VType[Float32])(holds).take(Axis[NodeClasses])(nodeClass)
    def placed(scores: Tensor2[Node, Pixel, V], carried: Tensor1[Node, Float32]) =
      Pixels.coordinates(scores.argmax(Axis[Pixel]), canvas) * carried
    val (drawn, runsOn) = (carries(_.isDrawn), carries(_.numPoints > 1))
    RecordNodes(
      nodeClass = nodeClass,
      startX = placed(logits.startX, drawn),
      startY = placed(logits.startY, drawn),
      endX = placed(logits.endX, runsOn),
      endY = placed(logits.endY, runsOn)
    )

object NodeScorer:

  /** A record's nodes, scored: a class, and a pixel for every coordinate a class can place. */
  case class NodeLogits[V](
      nodeClass: Tensor2[Node, NodeClasses, V],
      startX: Tensor2[Node, Pixel, V],
      startY: Tensor2[Node, Pixel, V],
      endX: Tensor2[Node, Pixel, V],
      endY: Tensor2[Node, Pixel, V]
  )

  case class Params[V](
      nodeClass: AffineLayer.Params[Embedding, NodeClasses, V],
      startX: AffineLayer.Params[Embedding, Pixel, V],
      startY: AffineLayer.Params[Embedding, Pixel, V],
      endX: AffineLayer.Params[Embedding, Pixel, V],
      endY: AffineLayer.Params[Embedding, Pixel, V]
  )

  object Params:
    given tensorTree: TensorTree[Params[Float32]] = TensorTree.derived
    given tree: TreeOf[Params[Float32], Float32] = TreeOf.derived

/** Reads an embedding per position back into a record's relationships, and settles them. */
class EdgeScorer[V: IsFloating](params: EdgeScorer.Params[V])
    extends (Tensor2[Edge, Embedding, V] => EdgeScorer.EdgeLogits[V]):

  import EdgeScorer.EdgeLogits

  private val classify = AffineLayer(params.edgeClass)

  override def apply(embeddings: Tensor2[Edge, Embedding, V]): EdgeLogits[V] =
    def named(end: AffineLayer.Params[Embedding, LinkedNode, V]) =
      embeddings.vmap(Axis[Edge])(AffineLayer(end))
    EdgeLogits(
      edgeClass = embeddings.vmap(Axis[Edge])(classify),
      subject = named(params.subject),
      obj = named(params.obj)
    )

  /** The scores settled. Settling on [[EdgeClass.NoEdge]] is where the relationships of a record
    * stop, and a position holding none relates nothing.
    */
  def decide(logits: EdgeLogits[V]): RecordEdges[Edge] =
    val edgeClass = logits.edgeClass.argmax(Axis[EdgeClasses])
    val relates = EdgeClass.indicator(VType[Float32])(_.relates).take(Axis[EdgeClasses])(edgeClass)
    def named(scores: Tensor2[Edge, LinkedNode, V]) =
      val end = scores.argmax(Axis[LinkedNode])
      where(relates > Tensor.like(relates).fill(0f), end, Tensor.like(end).fill(0))
    RecordEdges(edgeClass = edgeClass, subject = named(logits.subject), obj = named(logits.obj))

object EdgeScorer:

  /** A record's relationships, scored: a class, and a node for either end. */
  case class EdgeLogits[V](
      edgeClass: Tensor2[Edge, EdgeClasses, V],
      subject: Tensor2[Edge, LinkedNode, V],
      obj: Tensor2[Edge, LinkedNode, V]
  )

  case class Params[V](
      edgeClass: AffineLayer.Params[Embedding, EdgeClasses, V],
      subject: AffineLayer.Params[Embedding, LinkedNode, V],
      obj: AffineLayer.Params[Embedding, LinkedNode, V]
  )

  object Params:
    given tensorTree: TensorTree[Params[Float32]] = TensorTree.derived
    given tree: TreeOf[Params[Float32], Float32] = TreeOf.derived
