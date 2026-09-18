package d2g.model

import d2g.*
import d2g.train.*
import d2g.eval.*
import d2g.config.*
import dataset.EdgeClass
import dataset.EdgeClasses
import dataset.NodeClass
import dataset.NodeClasses
import dataset.RecordEdges
import dataset.RecordNodes
import deepwit.base.AffineLayer
import dimwit.*
import dimwit.Conversions.given

import scala.language.implicitConversions

trait NodePart derives Label // The parts a node embedding is composed of
trait EdgePart derives Label // The parts a edge embedding is composed of
trait PartEmbedding derives Label // An embedding of a [[NodePart]] or a [[EdgePart]]

/** Transforms a record node into a single embedding vector. */
class NodeEmbedder[V: IsFloating](params: NodeEmbedder.Params[V]) extends (RecordNodes[Node] => Tensor2[Node, Embedding, V]):

  private val project = AffineLayer(params.projection)

  /** How wide the canvas a coordinate is placed on is, i.e. how fine a pixel is. */
  private val canvas: Int = params.startX.shape(Axis[Pixel])

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

/** Transforms a record edge into a single embedding vector. */
class EdgeEmbedder[V: IsFloating](params: EdgeEmbedder.Params[V]) extends (RecordEdges[Edge] => Tensor2[Edge, Embedding, V]):

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
