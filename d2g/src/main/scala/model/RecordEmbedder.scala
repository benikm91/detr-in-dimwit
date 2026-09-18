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
import deepwit.embedder.VocabularyEmbedder
import dimwit.*
import dimwit.Conversions.given

import scala.language.implicitConversions

trait NodePart derives Label // The parts a node embedding is composed of
trait EdgePart derives Label // The parts a edge embedding is composed of
trait PartEmbedding derives Label // An embedding of a [[NodePart]] or a [[EdgePart]]

/** Transforms a record node into a single embedding vector. */
class NodeEmbedder[V: IsFloating](params: NodeEmbedder.Params[V]) extends (RecordNodes[Node] => Tensor2[Node, Embedding, V]):

  private val nodeClass = VocabularyEmbedder(params.nodeClass)

  // Coordinate vocabulary could be shared, yet we keep them separate just in case, preferring guaranteed capacity over slight speed gains.
  private val startX = VocabularyEmbedder(params.startX)
  private val startY = VocabularyEmbedder(params.startY)
  private val endX = VocabularyEmbedder(params.endX)
  private val endY = VocabularyEmbedder(params.endY)

  private val project = AffineLayer(params.projection)

  /** How wide the canvas a coordinate is placed on is, i.e. how fine a pixel is. */
  private val canvas: Int = params.startX.vocabularyEmbeddings.shape(Axis[Pixel])

  override def apply(nodes: RecordNodes[Node]): Tensor2[Node, Embedding, V] =
    def pixels(coordinate: Tensor1[Node, Float32]) = Pixels.of(coordinate, canvas)
    zipvmap(Axis[Node])(nodes.nodeClass, pixels(nodes.startX), pixels(nodes.startY), pixels(nodes.endX), pixels(nodes.endY)):
      case (cls, sx, sy, ex, ey) =>
        val parts = Seq(nodeClass(cls), startX(sx), startY(sy), endX(ex), endY(ey))
        project(stack(parts, Axis[NodePart]).flatten((Axis[NodePart], Axis[PartEmbedding])))

object NodeEmbedder:

  case class Params[V](
      nodeClass: VocabularyEmbedder.Params[NodeClasses, PartEmbedding, V],
      startX: VocabularyEmbedder.Params[Pixel, PartEmbedding, V],
      startY: VocabularyEmbedder.Params[Pixel, PartEmbedding, V],
      endX: VocabularyEmbedder.Params[Pixel, PartEmbedding, V],
      endY: VocabularyEmbedder.Params[Pixel, PartEmbedding, V],
      projection: AffineLayer.Params[NodePart |*| PartEmbedding, Embedding, V]
  )

  object Params:
    given tensorTree: TensorTree[Params[Float32]] = TensorTree.derived
    given tree: TreeOf[Params[Float32], Float32] = TreeOf.derived

/** Transforms a record edge into a single embedding vector. */
class EdgeEmbedder[V: IsFloating](params: EdgeEmbedder.Params[V]) extends (RecordEdges[Edge] => Tensor2[Edge, Embedding, V]):

  private val edgeClass = VocabularyEmbedder(params.edgeClass)
  private val subject = VocabularyEmbedder(params.subject)
  private val obj = VocabularyEmbedder(params.obj)
  private val project = AffineLayer(params.projection)

  override def apply(edges: RecordEdges[Edge]): Tensor2[Edge, Embedding, V] =
    zipvmap(Axis[Edge])(edges.edgeClass, edges.subject, edges.obj):
      case (cls, subj, ob) =>
        val parts = Seq(edgeClass(cls), subject(subj), obj(ob))
        project(stack(parts, Axis[EdgePart]).flatten((Axis[EdgePart], Axis[PartEmbedding])))

object EdgeEmbedder:

  case class Params[V](
      edgeClass: VocabularyEmbedder.Params[EdgeClasses, PartEmbedding, V],
      subject: VocabularyEmbedder.Params[LinkedNode, PartEmbedding, V],
      obj: VocabularyEmbedder.Params[LinkedNode, PartEmbedding, V],
      projection: AffineLayer.Params[EdgePart |*| PartEmbedding, Embedding, V]
  )

  object Params:
    given tensorTree: TensorTree[Params[Float32]] = TensorTree.derived
    given tree: TreeOf[Params[Float32], Float32] = TreeOf.derived
