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

/** Transforms an embedding back into a record node (logits). */
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
    val (drawn, runsOn) = (carries(_.isNode), carries(_.numPoints > 1))
    RecordNodes(
      nodeClass = nodeClass,
      startX = placed(logits.startX, drawn),
      startY = placed(logits.startY, drawn),
      endX = placed(logits.endX, runsOn),
      endY = placed(logits.endY, runsOn)
    )

object NodeHead:

  /** A record's nodes, scored: a class, and a pixel for every coordinate a class can place. */
  case class NodeLogits[V](
      nodeClass: Tensor2[Node, NodeClasses, V],
      startX: Tensor2[Node, Pixel, V],
      startY: Tensor2[Node, Pixel, V],
      endX: Tensor2[Node, Pixel, V],
      endY: Tensor2[Node, Pixel, V]
  )

  case class Params[V](
      nodeClass: AffineLayer.Params[Embedding, NodeClasses, V],
      startX: AffineLayer.Params[Embedding, Pixel, V],
      startY: AffineLayer.Params[Embedding, Pixel, V],
      endX: AffineLayer.Params[Embedding, Pixel, V],
      endY: AffineLayer.Params[Embedding, Pixel, V]
  )

  object Params:
    given tensorTree: TensorTree[Params[Float32]] = TensorTree.derived
    given tree: TreeOf[Params[Float32], Float32] = TreeOf.derived

/** Transforms an embedding back into a record edge (logits). */
class EdgeHead[V: IsFloating](params: EdgeHead.Params[V]) extends (Tensor2[Edge, Embedding, V] => EdgeHead.EdgeLogits[V]):

  import EdgeHead.EdgeLogits

  private val edgeClass = AffineLayer(params.edgeClass)
  private val subject = AffineLayer(params.subject)
  private val obj = AffineLayer(params.obj)

  override def apply(embeddings: Tensor2[Edge, Embedding, V]): EdgeLogits[V] =
    EdgeLogits(
      edgeClass = embeddings.vmap(Axis[Edge])(edgeClass),
      subject = embeddings.vmap(Axis[Edge])(subject),
      obj = embeddings.vmap(Axis[Edge])(obj)
    )

  /** The scores settled. Settling on [[EdgeClass.NoEdge]] is where the relationships of a record
    * stop, and a position holding none relates nothing.
    */
  def decide(logits: EdgeLogits[V]): RecordEdges[Edge] =
    val edgeClass = logits.edgeClass.argmax(Axis[EdgeClasses])
    val relates = !(edgeClass elementEquals_! EdgeClass.NoEdge.id)
    def named(scores: Tensor2[Edge, LinkedNode, V]) =
      val end = scores.argmax(Axis[LinkedNode])
      where_!(relates, end, 0)
    RecordEdges(edgeClass = edgeClass, subject = named(logits.subject), obj = named(logits.obj))

object EdgeHead:

  /** A record's relationships, scored: a class, and a node for either end. */
  case class EdgeLogits[V](
      edgeClass: Tensor2[Edge, EdgeClasses, V],
      subject: Tensor2[Edge, LinkedNode, V],
      obj: Tensor2[Edge, LinkedNode, V]
  )

  case class Params[V](
      edgeClass: AffineLayer.Params[Embedding, EdgeClasses, V],
      subject: AffineLayer.Params[Embedding, LinkedNode, V],
      obj: AffineLayer.Params[Embedding, LinkedNode, V]
  )

  object Params:
    given tensorTree: TensorTree[Params[Float32]] = TensorTree.derived
    given tree: TreeOf[Params[Float32], Float32] = TreeOf.derived
