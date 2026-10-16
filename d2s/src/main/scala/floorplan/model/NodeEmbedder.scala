package d2s.floorplan.model

import d2s.*
import d2s.model.*
import deepwit.base.AffineLayer
import deepwit.embedder.VocabularyEmbedder
import dimwit.*
import syntheticFloorplansDataset.FloorplanClasses
import syntheticFloorplansDataset.FloorplanNodes

import scala.language.implicitConversions

import NodeEmbedder.NodePart
import NodeEmbedder.PartEmbedding

/** Transforms a floor plan's node — a class and the two ends of a segment — into a single embedding
  * vector.
  */
class NodeEmbedder[V: IsFloating](params: NodeEmbedder.Params[V]) extends (FloorplanNodes[Node] => Tensor2[Node, Embedding, V]):

  private val nodeClass = VocabularyEmbedder(params.nodeClass)
  private val startX = VocabularyEmbedder(params.startX)
  private val startY = VocabularyEmbedder(params.startY)
  private val endX = VocabularyEmbedder(params.endX)
  private val endY = VocabularyEmbedder(params.endY)

  private val project = AffineLayer(params.projection)

  /** How wide the canvas a coordinate is placed on is, i.e. how fine a pixel is. */
  private val canvas: Int = params.startX.vocabularyEmbeddings.shape(Axis[Pixel])

  override def apply(nodes: FloorplanNodes[Node]): Tensor2[Node, Embedding, V] =
    def pixels(coordinate: Tensor1[Node, Float32]) = Pixels.of(coordinate, canvas)
    zipvmap(Axis[Node])(nodes.nodeClass, pixels(nodes.startX), pixels(nodes.startY), pixels(nodes.endX), pixels(nodes.endY)):
      case (cls, sx, sy, ex, ey) =>
        val parts = Seq(nodeClass(cls), startX(sx), startY(sy), endX(ex), endY(ey))
        project(stack(parts, Axis[NodePart]).flatten((Axis[NodePart], Axis[PartEmbedding])))

object NodeEmbedder:

  trait NodePart derives Label // The parts a node embedding is composed of
  trait PartEmbedding derives Label // An embedding of one part

  case class Params[V](
      nodeClass: VocabularyEmbedder.Params[FloorplanClasses, PartEmbedding, V],
      startX: VocabularyEmbedder.Params[Pixel, PartEmbedding, V],
      startY: VocabularyEmbedder.Params[Pixel, PartEmbedding, V],
      endX: VocabularyEmbedder.Params[Pixel, PartEmbedding, V],
      endY: VocabularyEmbedder.Params[Pixel, PartEmbedding, V],
      projection: AffineLayer.Params[NodePart |*| PartEmbedding, Embedding, V]
  )

  object Params:
    given tensorTree: TensorTree[Params[Float32]] = TensorTree.derived
    given tree: TreeOf[Params[Float32], Float32] = TreeOf.derived
