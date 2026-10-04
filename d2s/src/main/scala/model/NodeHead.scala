package d2s.model

import d2s.*
import dataset.EdgeClass
import dataset.EdgeClasses
import dataset.NodeClass
import dataset.NodeClasses
import dataset.RecordEdges
import dataset.RecordNodes
import deepwit.base.AffineLayer
import dimwit.*

import scala.language.implicitConversions

/** Transforms an embedding back into a record node (logits). */
class NodeHead[V: IsFloating](params: NodeHead.Params[V]) extends (Tensor2[Node, Embedding, V] => NodeHead.NodeLogits[V]):

  import NodeHead.NodeLogits

  private val nodeClass = AffineLayer(params.nodeClass)
  private val startX = AffineLayer(params.startX)
  private val startY = AffineLayer(params.startY)
  private val endX = AffineLayer(params.endX)
  private val endY = AffineLayer(params.endY)
  private val midX = AffineLayer(params.midX)
  private val midY = AffineLayer(params.midY)

  val canvas: Int = params.startX.bias.shape(Axis[Pixel])

  override def apply(embeddings: Tensor2[Node, Embedding, V]): NodeLogits[V] =
    NodeLogits(
      nodeClass = embeddings.vmap(Axis[Node])(nodeClass),
      startX = embeddings.vmap(Axis[Node])(startX),
      startY = embeddings.vmap(Axis[Node])(startY),
      endX = embeddings.vmap(Axis[Node])(endX),
      endY = embeddings.vmap(Axis[Node])(endY),
      midX = embeddings.vmap(Axis[Node])(midX),
      midY = embeddings.vmap(Axis[Node])(midY)
    )

  /** The scores settled. Settling on [[NodeClass.NoNode]] is where the nodes of a record stop.
    *
    * A node is placed only where its class places itself, so the rest is cleared rather than left
    * at whatever an unsupervised head happened to say — a node the model takes has to look like a
    * node the data would have given it.
    */
  def decide(logits: NodeLogits[V]): RecordNodes[Node] =
    val nodeClass = logits.nodeClass.argmax(Axis[NodeClasses])
    def carries(holds: NodeClass => Boolean) = NodeClass.indicator(VType[Float32])(holds).take(Axis[NodeClasses])(nodeClass)
    def placed(scores: Tensor2[Node, Pixel, V], carried: Tensor1[Node, Float32]) =
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

  /** A record's nodes, scored: a class, and a pixel for every coordinate a class can place. */
  case class NodeLogits[V](
      nodeClass: Tensor2[Node, NodeClasses, V],
      startX: Tensor2[Node, Pixel, V],
      startY: Tensor2[Node, Pixel, V],
      endX: Tensor2[Node, Pixel, V],
      endY: Tensor2[Node, Pixel, V],
      midX: Tensor2[Node, Pixel, V],
      midY: Tensor2[Node, Pixel, V]
  )

  case class Params[V](
      nodeClass: AffineLayer.Params[Embedding, NodeClasses, V],
      startX: AffineLayer.Params[Embedding, Pixel, V],
      startY: AffineLayer.Params[Embedding, Pixel, V],
      endX: AffineLayer.Params[Embedding, Pixel, V],
      endY: AffineLayer.Params[Embedding, Pixel, V],
      midX: AffineLayer.Params[Embedding, Pixel, V],
      midY: AffineLayer.Params[Embedding, Pixel, V]
  )

  object Params:
    given tensorTree: TensorTree[Params[Float32]] = TensorTree.derived
    given tree: TreeOf[Params[Float32], Float32] = TreeOf.derived
