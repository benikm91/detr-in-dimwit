package detrbucket.sketch.model

import detr.*
import detrbucket.sketch.Buckets
import dataset.IsConstruction
import dataset.NodeClass
import dataset.NodeClasses
import dataset.RecordNodes
import deepwit.base.AffineLayer
import dimwit.*

import scala.language.implicitConversions

/** Whether a query answers with a node at all: 0 for none, 1 for one of its bucket's class. */
trait Presence derives Label

/** Transforms the embedding of every query into a record node (logits). A query does not choose
  * its class, its bucket does; it says only whether it answers, and where.
  */
class BucketHead[V: IsFloating](params: BucketHead.Params[V], buckets: Buckets) extends (Tensor2[Query, DETRBucket.Embedding, V] => BucketHead.NodeLogits[V]):

  import BucketHead.NodeLogits

  private val presence = AffineLayer(params.presence)
  private val construction = AffineLayer(params.construction)
  private val startX = AffineLayer(params.startX)
  private val startY = AffineLayer(params.startY)
  private val endX = AffineLayer(params.endX)
  private val endY = AffineLayer(params.endY)
  private val midX = AffineLayer(params.midX)
  private val midY = AffineLayer(params.midY)

  val canvas: Int = params.startX.bias.shape(Axis[Pixel])

  override def apply(embeddings: Tensor2[Query, DETRBucket.Embedding, V]): NodeLogits[V] =
    NodeLogits(
      presence = embeddings.vmap(Axis[Query])(presence),
      construction = embeddings.vmap(Axis[Query])(construction),
      startX = embeddings.vmap(Axis[Query])(startX),
      startY = embeddings.vmap(Axis[Query])(startY),
      endX = embeddings.vmap(Axis[Query])(endX),
      endY = embeddings.vmap(Axis[Query])(endY),
      midX = embeddings.vmap(Axis[Query])(midX),
      midY = embeddings.vmap(Axis[Query])(midY)
    )

  /** The scores settled: a query that answers takes its bucket's class, one that does not answers
    * with [[NodeClass.NoNode]]. A node is placed only where its class places itself.
    */
  def decide(logits: NodeLogits[V]): RecordNodes[Query] =
    val answers = logits.presence.argmax(Axis[Presence]) elementEquals_! 1
    val nodeClass = where_!(answers, buckets.classes, NodeClass.NoNode.id)
    def carries(holds: NodeClass => Boolean) = NodeClass.indicator(VType[Float32])(holds).slice(Axis[NodeClasses].at(nodeClass))
    def placed(scores: Tensor2[Query, Pixel, V], carried: Tensor1[Query, Float32]) =
      Pixels.coordinates(scores.argmax(Axis[Pixel]), canvas) * carried
    val (drawn, runsOn, bends) = (carries(_.isNode), carries(_.numPoints > 1), carries(_.numPoints > 2))
    RecordNodes(
      nodeClass = nodeClass,
      construction = logits.construction.argmax(Axis[IsConstruction]) * drawn.asInt(VType[Int32]),
      startX = placed(logits.startX, drawn),
      startY = placed(logits.startY, drawn),
      endX = placed(logits.endX, runsOn),
      endY = placed(logits.endY, runsOn),
      midX = placed(logits.midX, bends),
      midY = placed(logits.midY, bends)
    )

object BucketHead:

  /** Every query's node, scored: whether it answers, whether it is construction geometry, and a pixel
    * for every coordinate a class can place.
    */
  case class NodeLogits[V](
      presence: Tensor2[Query, Presence, V],
      construction: Tensor2[Query, IsConstruction, V],
      startX: Tensor2[Query, Pixel, V],
      startY: Tensor2[Query, Pixel, V],
      endX: Tensor2[Query, Pixel, V],
      endY: Tensor2[Query, Pixel, V],
      midX: Tensor2[Query, Pixel, V],
      midY: Tensor2[Query, Pixel, V]
  )

  object NodeLogits:

    given tensorTree[V]: TensorTree[NodeLogits[V]] = TensorTree.derived
    given tree[V]: TreeOf[NodeLogits[V], V] = TreeOf.derived

  case class Params[V](
      presence: AffineLayer.Params[DETRBucket.Embedding, Presence, V],
      construction: AffineLayer.Params[DETRBucket.Embedding, IsConstruction, V],
      startX: AffineLayer.Params[DETRBucket.Embedding, Pixel, V],
      startY: AffineLayer.Params[DETRBucket.Embedding, Pixel, V],
      endX: AffineLayer.Params[DETRBucket.Embedding, Pixel, V],
      endY: AffineLayer.Params[DETRBucket.Embedding, Pixel, V],
      midX: AffineLayer.Params[DETRBucket.Embedding, Pixel, V],
      midY: AffineLayer.Params[DETRBucket.Embedding, Pixel, V]
  )

  object Params:
    given tensorTree: TensorTree[Params[Float32]] = TensorTree.derived
    given tree: TreeOf[Params[Float32], Float32] = TreeOf.derived
