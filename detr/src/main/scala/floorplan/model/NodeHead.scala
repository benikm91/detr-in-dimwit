package detr.floorplan.model

import detr.*
import deepwit.base.AffineLayer
import dimwit.*
import syntheticFloorplansDataset.FloorplanClass
import syntheticFloorplansDataset.FloorplanClasses
import syntheticFloorplansDataset.FloorplanNodes

import scala.language.implicitConversions

/** Transforms the embedding of every query into a floor plan's node (logits). */
class NodeHead[V: IsFloating](params: NodeHead.Params[V]) extends (Tensor2[Query, DETR.Embedding, V] => NodeHead.NodeLogits[V]):

  import NodeHead.NodeLogits

  private val nodeClass = AffineLayer(params.nodeClass)
  private val startX = AffineLayer(params.startX)
  private val startY = AffineLayer(params.startY)
  private val endX = AffineLayer(params.endX)
  private val endY = AffineLayer(params.endY)

  val canvas: Int = params.startX.bias.shape(Axis[Pixel])

  override def apply(embeddings: Tensor2[Query, DETR.Embedding, V]): NodeLogits[V] =
    NodeLogits(
      nodeClass = embeddings.vmap(Axis[Query])(nodeClass),
      startX = embeddings.vmap(Axis[Query])(startX),
      startY = embeddings.vmap(Axis[Query])(startY),
      endX = embeddings.vmap(Axis[Query])(endX),
      endY = embeddings.vmap(Axis[Query])(endY)
    )

  /** The scores settled. A query settling on [[FloorplanClass.NoNode]] answers with no node, which
    * is placed nowhere.
    */
  def decide(logits: NodeLogits[V]): FloorplanNodes[Query] =
    val nodeClass = logits.nodeClass.argmax(Axis[FloorplanClasses])
    val drawn = FloorplanClass.indicator(VType[Float32])(_.isNode).slice(Axis[FloorplanClasses].at(nodeClass))
    def placed(scores: Tensor2[Query, Pixel, V]) = Pixels.coordinates(scores.argmax(Axis[Pixel]), canvas) * drawn
    FloorplanNodes(nodeClass, placed(logits.startX), placed(logits.startY), placed(logits.endX), placed(logits.endY))

object NodeHead:

  /** Every query's node, scored: a class, and a pixel for each coordinate of its two ends. */
  case class NodeLogits[V](
      nodeClass: Tensor2[Query, FloorplanClasses, V],
      startX: Tensor2[Query, Pixel, V],
      startY: Tensor2[Query, Pixel, V],
      endX: Tensor2[Query, Pixel, V],
      endY: Tensor2[Query, Pixel, V]
  )

  object NodeLogits:

    given tensorTree[V]: TensorTree[NodeLogits[V]] = TensorTree.derived
    given tree[V]: TreeOf[NodeLogits[V], V] = TreeOf.derived

  case class Params[V](
      nodeClass: AffineLayer.Params[DETR.Embedding, FloorplanClasses, V],
      startX: AffineLayer.Params[DETR.Embedding, Pixel, V],
      startY: AffineLayer.Params[DETR.Embedding, Pixel, V],
      endX: AffineLayer.Params[DETR.Embedding, Pixel, V],
      endY: AffineLayer.Params[DETR.Embedding, Pixel, V]
  )

  object Params:
    given tensorTree: TensorTree[Params[Float32]] = TensorTree.derived
    given tree: TreeOf[Params[Float32], Float32] = TreeOf.derived
