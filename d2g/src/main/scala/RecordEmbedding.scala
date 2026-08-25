import dataset.EdgeClass
import dataset.EdgeClasses
import dataset.NodeClass
import dataset.NodeClasses
import dataset.NodeLink
import dataset.NodePoint
import dataset.RecordEdges
import dataset.RecordNodes
import deepwit.base.AffineLayer
import dimwit.*

/** One linear map per value a position carries: a head of its own, a vocabulary shared. */
case class ScorerHead[Carries, Values, V](
    projection: Tensor3[Carries, Embedding, Values, V],
    bias: Tensor2[Carries, Values, V]
)

/** Reads a record's nodes into one embedding each: the class and the points it is placed by
  * embedded on their own, concatenated and projected into the space the decoder works in.
  */
class NodeEmbedder[V: IsFloating](params: NodeEmbedder.Params[V], canvas: Int)
    extends (RecordNodes[Node] => Tensor2[Node, Embedding, V]):

  private val project = AffineLayer(params.projection)

  override def apply(nodes: RecordNodes[Node]): Tensor2[Node, Embedding, V] =
    val parts = params.nodeClass.take(Axis[NodeClasses])(nodes.nodeClass) +:
      (placed(params.xs, Pixels.of(nodes.xs, canvas)) ++ placed(params.ys, Pixels.of(nodes.ys, canvas)))
    stack(parts, Axis[NodePart])
      .swap(Axis[NodePart], Axis[Node])
      .flatten((Axis[NodePart], Axis[PartEmbedding]))
      .vmap(Axis[Node])(project)

  /** One embedding per point, looked up in the table that point is read by. */
  private def placed(
      tables: Tensor3[NodePoint, Pixel, PartEmbedding, V],
      pixels: Tensor2[Node, NodePoint, Int32]
  ): Seq[Tensor2[Node, PartEmbedding, V]] =
    tables.unstack(Axis[NodePoint]).zip(pixels.unstack(Axis[NodePoint])).map((table, pixel) => table.take(Axis[Pixel])(pixel))

object NodeEmbedder:

  case class Params[V](
      nodeClass: Tensor2[NodeClasses, PartEmbedding, V],
      xs: Tensor3[NodePoint, Pixel, PartEmbedding, V],
      ys: Tensor3[NodePoint, Pixel, PartEmbedding, V],
      projection: AffineLayer.Params[NodePart |*| PartEmbedding, Embedding, V]
  )

  object Params:
    given tensorTree: TensorTree[Params[Float32]] = TensorTree.derived
    given tree: TreeOf[Params[Float32], Float32] = TreeOf.derived

/** Reads a record's relationships into one embedding each: the class and the nodes it links
  * embedded on their own, concatenated and projected into the space the decoder works in.
  */
class EdgeEmbedder[V: IsFloating](params: EdgeEmbedder.Params[V])
    extends (RecordEdges[Edge] => Tensor2[Edge, Embedding, V]):

  private val project = AffineLayer(params.projection)

  override def apply(edges: RecordEdges[Edge]): Tensor2[Edge, Embedding, V] =
    val parts = params.edgeClass.take(Axis[EdgeClasses])(edges.edgeClass) +: linked(edges.links)
    stack(parts, Axis[EdgePart])
      .swap(Axis[EdgePart], Axis[Edge])
      .flatten((Axis[EdgePart], Axis[PartEmbedding]))
      .vmap(Axis[Edge])(project)

  /** One embedding per end, looked up in the table that end is read by. */
  private def linked(links: Tensor2[Edge, NodeLink, Int32]): Seq[Tensor2[Edge, PartEmbedding, V]] =
    params.links.unstack(Axis[NodeLink]).zip(links.unstack(Axis[NodeLink])).map((table, named) => table.take(Axis[LinkedNode])(named))

object EdgeEmbedder:

  case class Params[V](
      edgeClass: Tensor2[EdgeClasses, PartEmbedding, V],
      links: Tensor3[NodeLink, LinkedNode, PartEmbedding, V],
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
  val canvas: Int = params.xs.bias.shape(Axis[Pixel])

  override def apply(embeddings: Tensor2[Node, Embedding, V]): NodeLogits[V] =
    def scored(head: ScorerHead[NodePoint, Pixel, V]) =
      embeddings.vmap(Axis[Node])(embedding => embedding.dot(Axis[Embedding])(head.projection) + head.bias)
    NodeLogits(
      nodeClass = embeddings.vmap(Axis[Node])(classify),
      xs = scored(params.xs),
      ys = scored(params.ys)
    )

  /** The scores settled. Settling on [[NodeClass.NoNode]] is where the nodes of a record stop.
    *
    * A node carries only what its class carries, so the rest is cleared rather than left at
    * whatever an unsupervised head happened to say — a node the model takes has to look like a
    * node the data would have given it.
    */
  def decide(logits: NodeLogits[V]): RecordNodes[Node] =
    val nodeClass = logits.nodeClass.argmax(Axis[NodeClasses])
    val used = NodeClass.usedPoints(VType[Float32]).take(Axis[NodeClasses])(nodeClass)
    def coordinates(scores: Tensor3[Node, NodePoint, Pixel, V]) =
      Pixels.coordinates(scores.argmax(Axis[Pixel]), canvas) * used
    RecordNodes(nodeClass = nodeClass, xs = coordinates(logits.xs), ys = coordinates(logits.ys))

object NodeScorer:

  /** A record's nodes, scored: a class and a pixel per coordinate, per position. */
  case class NodeLogits[V](
      nodeClass: Tensor2[Node, NodeClasses, V],
      xs: Tensor3[Node, NodePoint, Pixel, V],
      ys: Tensor3[Node, NodePoint, Pixel, V]
  )

  case class Params[V](
      nodeClass: AffineLayer.Params[Embedding, NodeClasses, V],
      xs: ScorerHead[NodePoint, Pixel, V],
      ys: ScorerHead[NodePoint, Pixel, V]
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
    EdgeLogits(
      edgeClass = embeddings.vmap(Axis[Edge])(classify),
      links = embeddings.vmap(Axis[Edge])(embedding => embedding.dot(Axis[Embedding])(params.links.projection) + params.links.bias)
    )

  /** The scores settled. Settling on [[EdgeClass.NoEdge]] is where the relationships of a record
    * stop, and a position holding none links nothing.
    */
  def decide(logits: EdgeLogits[V]): RecordEdges[Edge] =
    val edgeClass = logits.edgeClass.argmax(Axis[EdgeClasses])
    val used = EdgeClass.usedLinks(VType[Float32]).take(Axis[EdgeClasses])(edgeClass)
    val named = logits.links.argmax(Axis[LinkedNode])
    RecordEdges(edgeClass = edgeClass, links = where(used > Tensor.like(used).fill(0f), named, Tensor.like(named).fill(0)))

object EdgeScorer:

  /** A record's relationships, scored: a class and a node per link, per position. */
  case class EdgeLogits[V](
      edgeClass: Tensor2[Edge, EdgeClasses, V],
      links: Tensor3[Edge, NodeLink, LinkedNode, V]
  )

  case class Params[V](
      edgeClass: AffineLayer.Params[Embedding, EdgeClasses, V],
      links: ScorerHead[NodeLink, LinkedNode, V]
  )

  object Params:
    given tensorTree: TensorTree[Params[Float32]] = TensorTree.derived
    given tree: TreeOf[Params[Float32], Float32] = TreeOf.derived
