import dataset.NodeClass
import dataset.NodeClasses
import dataset.NodeLink
import dataset.NodePoint
import dataset.Record
import deepwit.base.AffineLayer
import dimwit.*

/** A record's worth of scores: a class, a pixel per coordinate, a node per link. */
case class NodeLogits[V](
    nodeClass: Tensor2[Node, NodeClasses, V],
    xs: Tensor3[Node, NodePoint, Pixel, V],
    ys: Tensor3[Node, NodePoint, Pixel, V],
    links: Tensor3[Node, NodeLink, LinkedNode, V]
)

/** Reads a record into one embedding per node: the class and everything the node carries embedded
  * on their own, concatenated and projected into the space the decoder works in.
  */
class NodeEmbedder[V: IsFloating](params: NodeEmbedder.Params[V], canvas: Int)
    extends (Record[Node] => Tensor2[Node, Embedding, V]):

  private val project = AffineLayer(params.projection)

  override def apply(record: Record[Node]): Tensor2[Node, Embedding, V] =
    val parts = params.nodeClass.take(Axis[NodeClasses])(record.nodeClass) +:
      (embedded(params.xs, Axis[NodePoint], Pixels.of(record.xs, canvas)) ++
        embedded(params.ys, Axis[NodePoint], Pixels.of(record.ys, canvas)) ++
        embedded(params.links, Axis[NodeLink], record.links))
    stack(parts, Axis[NodePart])
      .swap(Axis[NodePart], Axis[Node])
      .flatten((Axis[NodePart], Axis[PartEmbedding]))
      .vmap(Axis[Node])(project)

  /** One embedding per carried value, looked up in the table that value is read by. */
  private def embedded[Carries: Label, Values: Label](
      tables: Tensor3[Carries, Values, PartEmbedding, V],
      carries: Axis[Carries],
      values: Tensor2[Node, Carries, Int32]
  ): Seq[Tensor2[Node, PartEmbedding, V]] =
    tables.unstack(carries).zip(values.unstack(carries)).map((table, value) => table.take(Axis[Values])(value))

object NodeEmbedder:

  case class Params[V](
      nodeClass: Tensor2[NodeClasses, PartEmbedding, V],
      xs: Tensor3[NodePoint, Pixel, PartEmbedding, V],
      ys: Tensor3[NodePoint, Pixel, PartEmbedding, V],
      links: Tensor3[NodeLink, LinkedNode, PartEmbedding, V],
      projection: AffineLayer.Params[NodePart |*| PartEmbedding, Embedding, V]
  )

  object Params:
    given tensorTree: TensorTree[Params[Float32]] = TensorTree.derived
    given tree: TreeOf[Params[Float32], Float32] = TreeOf.derived

/** Reads an embedding per node back into a record's worth of scores, and settles them. */
class NodeScorer[V: IsFloating](params: NodeScorer.Params[V])
    extends (Tensor2[Node, Embedding, V] => NodeLogits[V]):

  private val classify = AffineLayer(params.nodeClass)

  /** How wide the canvas a coordinate is scored on is, i.e. how fine a pixel is. */
  val canvas: Int = params.xs.bias.shape(Axis[Pixel])

  override def apply(embeddings: Tensor2[Node, Embedding, V]): NodeLogits[V] =
    def scored[Carries: Label, Values: Label](head: NodeScorer.Head[Carries, Values, V]) =
      embeddings.vmap(Axis[Node])(embedding => embedding.dot(Axis[Embedding])(head.projection) + head.bias)
    NodeLogits(
      nodeClass = embeddings.vmap(Axis[Node])(classify),
      xs = scored(params.xs),
      ys = scored(params.ys),
      links = scored(params.links)
    )

  /** The scores settled. Settling on [[NodeClass.NoNode]] is where the record stops.
    *
    * A node carries only what its class carries, so the rest is cleared rather than left at
    * whatever an unsupervised head happened to say — a node the model takes has to look like a
    * node the data would have given it.
    */
  def decide(logits: NodeLogits[V]): Record[Node] =
    val nodeClass = logits.nodeClass.argmax(Axis[NodeClasses])
    def carried[Carries: Label](used: Tensor2[NodeClasses, Carries, Float32]) =
      used.take(Axis[NodeClasses])(nodeClass)
    def coordinates(scores: Tensor3[Node, NodePoint, Pixel, V]) =
      Pixels.coordinates(scores.argmax(Axis[Pixel]), canvas) * carried(NodeClass.usedPoints(VType[Float32]))
    val links = logits.links.argmax(Axis[LinkedNode])
    Record(
      nodeClass = nodeClass,
      xs = coordinates(logits.xs),
      ys = coordinates(logits.ys),
      links = where(
        carried(NodeClass.usedLinks(VType[Float32])) > Tensor.like(links).fill(0).asFloat(VType[Float32]),
        links,
        Tensor.like(links).fill(0)
      )
    )

object NodeScorer:

  /** One linear map per value a node carries: a head of its own, a vocabulary shared. */
  case class Head[Carries, Values, V](
      projection: Tensor3[Carries, Embedding, Values, V],
      bias: Tensor2[Carries, Values, V]
  )

  case class Params[V](
      nodeClass: AffineLayer.Params[Embedding, NodeClasses, V],
      xs: Head[NodePoint, Pixel, V],
      ys: Head[NodePoint, Pixel, V],
      links: Head[NodeLink, LinkedNode, V]
  )

  object Params:
    given tensorTree: TensorTree[Params[Float32]] = TensorTree.derived
    given tree: TreeOf[Params[Float32], Float32] = TreeOf.derived
