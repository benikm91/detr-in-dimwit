package detr.model

import detr.*
import dataset.NodeClass
import dataset.NodeClasses
import dataset.RecordNodes
import deepwit.base.AffineLayer
import dimwit.*

import scala.language.implicitConversions

/** Transforms the embedding of every query into a record node (logits), the way the transcriber's
  * node head does.
  */
class NodeHead[V: IsFloating](params: NodeHead.Params[V]) extends (Tensor2[Query, DETR.Embedding, V] => NodeHead.NodeLogits[V]):

  import NodeHead.NodeLogits

  private val nodeClass = AffineLayer(params.nodeClass)
  private val startX = AffineLayer(params.startX)
  private val startY = AffineLayer(params.startY)
  private val endX = AffineLayer(params.endX)
  private val endY = AffineLayer(params.endY)
  private val midX = AffineLayer(params.midX)
  private val midY = AffineLayer(params.midY)

  val canvas: Int = params.startX.bias.shape(Axis[Pixel])

  override def apply(embeddings: Tensor2[Query, DETR.Embedding, V]): NodeLogits[V] =
    NodeLogits(
      nodeClass = embeddings.vmap(Axis[Query])(nodeClass),
      startX = embeddings.vmap(Axis[Query])(startX),
      startY = embeddings.vmap(Axis[Query])(startY),
      endX = embeddings.vmap(Axis[Query])(endX),
      endY = embeddings.vmap(Axis[Query])(endY),
      midX = embeddings.vmap(Axis[Query])(midX),
      midY = embeddings.vmap(Axis[Query])(midY)
    )

  /** The scores settled. A query settling on [[NodeClass.NoNode]] answers with no node.
    *
    * A node is placed only where its class places itself, so the rest is cleared rather than left
    * at whatever an unsupervised head happened to say — a node the model takes has to look like a
    * node the data would have given it.
    */
  def decide(logits: NodeLogits[V]): RecordNodes[Query] =
    val nodeClass = logits.nodeClass.argmax(Axis[NodeClasses])
    def carries(holds: NodeClass => Boolean) = NodeClass.indicator(VType[Float32])(holds).take(Axis[NodeClasses])(nodeClass)
    def placed(scores: Tensor2[Query, Pixel, V], carried: Tensor1[Query, Float32]) =
      Pixels.coordinates(scores.argmax(Axis[Pixel]), canvas) * carried
    val (drawn, runsOn, bends) = (carries(_.isNode), carries(_.numPoints > 1), carries(_.numPoints > 2))
    RecordNodes(
      nodeClass = nodeClass,
      startX = placed(logits.startX, drawn),
      startY = placed(logits.startY, drawn),
      endX = placed(logits.endX, runsOn),
      endY = placed(logits.endY, runsOn),
      midX = placed(logits.midX, bends),
      midY = placed(logits.midY, bends)
    )

object NodeHead:

  /** Every query's node, scored: a class, and a pixel for every coordinate a class can place. */
  case class NodeLogits[V](
      nodeClass: Tensor2[Query, NodeClasses, V],
      startX: Tensor2[Query, Pixel, V],
      startY: Tensor2[Query, Pixel, V],
      endX: Tensor2[Query, Pixel, V],
      endY: Tensor2[Query, Pixel, V],
      midX: Tensor2[Query, Pixel, V],
      midY: Tensor2[Query, Pixel, V]
  )

  case class Params[V](
      nodeClass: AffineLayer.Params[DETR.Embedding, NodeClasses, V],
      startX: AffineLayer.Params[DETR.Embedding, Pixel, V],
      startY: AffineLayer.Params[DETR.Embedding, Pixel, V],
      endX: AffineLayer.Params[DETR.Embedding, Pixel, V],
      endY: AffineLayer.Params[DETR.Embedding, Pixel, V],
      midX: AffineLayer.Params[DETR.Embedding, Pixel, V],
      midY: AffineLayer.Params[DETR.Embedding, Pixel, V]
  )

  object Params:
    given tensorTree: TensorTree[Params[Float32]] = TensorTree.derived
    given tree: TreeOf[Params[Float32], Float32] = TreeOf.derived
