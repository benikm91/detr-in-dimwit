package d2s.model

import d2s.*
import dataset.EdgeClass
import dataset.EdgeClasses
import dataset.NodeClass
import dataset.NodeClasses
import dataset.IsConstruction
import dataset.RecordEdges
import dataset.RecordNodes
import deepwit.base.AffineLayer
import deepwit.embedder.VocabularyEmbedder
import dimwit.*

import scala.language.implicitConversions

trait NodePart derives Label // The parts a node embedding is composed of
trait PartEmbedding derives Label // An embedding of a [[NodePart]] or a [[EdgePart]]

/** Transforms a record node into a single embedding vector. */
class NodeEmbedder[V: IsFloating](params: NodeEmbedder.Params[V]) extends (RecordNodes[Node] => Tensor2[Node, Embedding, V]):

  private val nodeClass = VocabularyEmbedder(params.nodeClass)
  private val construction = VocabularyEmbedder(params.construction)

  // Coordinate vocabulary could be shared, yet we keep them separate just in case, preferring guaranteed capacity over slight speed gains.
  private val startX = VocabularyEmbedder(params.startX)
  private val startY = VocabularyEmbedder(params.startY)
  private val endX = VocabularyEmbedder(params.endX)
  private val endY = VocabularyEmbedder(params.endY)
  private val midX = VocabularyEmbedder(params.midX)
  private val midY = VocabularyEmbedder(params.midY)

  private val project = AffineLayer(params.projection)

  /** How wide the canvas a coordinate is placed on is, i.e. how fine a pixel is. */
  private val canvas: Int = params.startX.vocabularyEmbeddings.shape(Axis[Pixel])

  override def apply(nodes: RecordNodes[Node]): Tensor2[Node, Embedding, V] =
    def pixels(coordinate: Tensor1[Node, Float32]) = Pixels.of(coordinate, canvas)
    zipvmap(Axis[Node])(nodes.nodeClass, nodes.construction, pixels(nodes.startX), pixels(nodes.startY), pixels(nodes.endX), pixels(nodes.endY), pixels(nodes.midX), pixels(nodes.midY)):
      case (cls, isConstruction, sx, sy, ex, ey, mx, my) =>
        val parts = Seq(nodeClass(cls), construction(isConstruction), startX(sx), startY(sy), endX(ex), endY(ey), midX(mx), midY(my))
        project(stack(parts, Axis[NodePart]).flatten((Axis[NodePart], Axis[PartEmbedding])))

object NodeEmbedder:

  case class Params[V](
      nodeClass: VocabularyEmbedder.Params[NodeClasses, PartEmbedding, V],
      construction: VocabularyEmbedder.Params[IsConstruction, PartEmbedding, V],
      startX: VocabularyEmbedder.Params[Pixel, PartEmbedding, V],
      startY: VocabularyEmbedder.Params[Pixel, PartEmbedding, V],
      endX: VocabularyEmbedder.Params[Pixel, PartEmbedding, V],
      endY: VocabularyEmbedder.Params[Pixel, PartEmbedding, V],
      midX: VocabularyEmbedder.Params[Pixel, PartEmbedding, V],
      midY: VocabularyEmbedder.Params[Pixel, PartEmbedding, V],
      projection: AffineLayer.Params[NodePart |*| PartEmbedding, Embedding, V]
  )

  object Params:
    given tensorTree: TensorTree[Params[Float32]] = TensorTree.derived
    given tree: TreeOf[Params[Float32], Float32] = TreeOf.derived
