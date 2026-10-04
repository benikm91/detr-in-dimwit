package d2g.model

import d2g.*
import d2g.train.*
import d2g.eval.*
import d2g.config.*
import d2s.model.*
import dataset.EdgeClass
import dataset.EdgeClasses
import dataset.NodeClass
import dataset.NodeClasses
import dataset.RecordEdges
import dataset.RecordNodes
import deepwit.base.AffineLayer
import dimwit.*

import scala.language.implicitConversions

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
