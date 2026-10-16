package d2s.floorplan.model

import d2s.*
import d2s.model.*
import deepwit.base.AffineLayer
import dimwit.*
import syntheticFloorplansDataset.FloorplanClass
import syntheticFloorplansDataset.FloorplanClasses
import syntheticFloorplansDataset.FloorplanNodes

import scala.language.implicitConversions

/** Transforms an embedding back into a floor plan's node (logits). */
class NodeHead[V: IsFloating](params: NodeHead.Params[V]) extends (Tensor2[Node, Embedding, V] => NodeHead.NodeLogits[V]):

  import NodeHead.NodeLogits

  private val nodeClass = AffineLayer(params.nodeClass)
  private val startX = AffineLayer(params.startX)
  private val startY = AffineLayer(params.startY)
  private val endX = AffineLayer(params.endX)
  private val endY = AffineLayer(params.endY)

  val canvas: Int = params.startX.bias.shape(Axis[Pixel])

  override def apply(embeddings: Tensor2[Node, Embedding, V]): NodeLogits[V] =
    NodeLogits(
      nodeClass = embeddings.vmap(Axis[Node])(nodeClass),
      startX = embeddings.vmap(Axis[Node])(startX),
      startY = embeddings.vmap(Axis[Node])(startY),
      endX = embeddings.vmap(Axis[Node])(endX),
      endY = embeddings.vmap(Axis[Node])(endY)
    )

  /** The scores settled. Settling on [[FloorplanClass.NoNode]] is where the nodes of a record stop,
    * and a node that is not there is placed nowhere.
    */
  def decide(logits: NodeLogits[V]): FloorplanNodes[Node] =
    val nodeClass = logits.nodeClass.argmax(Axis[FloorplanClasses])
    val drawn = FloorplanClass.indicator(VType[Float32])(_.isNode).slice(Axis[FloorplanClasses].at(nodeClass))
    def placed(scores: Tensor2[Node, Pixel, V]) = Pixels.coordinates(scores.argmax(Axis[Pixel]), canvas) * drawn
    FloorplanNodes(nodeClass, placed(logits.startX), placed(logits.startY), placed(logits.endX), placed(logits.endY))

object NodeHead:

  /** A floor plan's nodes, scored: a class, and a pixel for each coordinate of its two ends. */
  case class NodeLogits[V](
      nodeClass: Tensor2[Node, FloorplanClasses, V],
      startX: Tensor2[Node, Pixel, V],
      startY: Tensor2[Node, Pixel, V],
      endX: Tensor2[Node, Pixel, V],
      endY: Tensor2[Node, Pixel, V]
  )

  case class Params[V](
      nodeClass: AffineLayer.Params[Embedding, FloorplanClasses, V],
      startX: AffineLayer.Params[Embedding, Pixel, V],
      startY: AffineLayer.Params[Embedding, Pixel, V],
      endX: AffineLayer.Params[Embedding, Pixel, V],
      endY: AffineLayer.Params[Embedding, Pixel, V]
  )

  object Params:
    given tensorTree: TensorTree[Params[Float32]] = TensorTree.derived
    given tree: TreeOf[Params[Float32], Float32] = TreeOf.derived
