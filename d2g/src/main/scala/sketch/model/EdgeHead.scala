package d2g.sketch.model

import d2g.*
import d2g.model.*
import dataset.EdgeClass
import dataset.EdgeClasses
import dataset.RecordEdges
import deepwit.base.AffineLayer
import dimwit.*

import scala.language.implicitConversions

/** Transforms an embedding back into a drawing's relationship (logits): what it is, and a
  * [[Pointer]] at each of the two nodes it relates.
  */
class EdgeHead[V: IsFloating](params: EdgeHead.Params[V]):

  import EdgeHead.EdgeLogits

  private val edgeClass = AffineLayer(params.edgeClass)
  private val subject = Pointer(params.subject)
  private val obj = Pointer(params.obj)

  def apply(embeddings: Tensor2[Edge, Embedding, V], nodes: NodeSource[Embedding, V]): EdgeLogits[V] =
    EdgeLogits(
      edgeClass = embeddings.vmap(Axis[Edge])(edgeClass),
      subject = subject(embeddings, nodes),
      obj = obj(embeddings, nodes)
    )

  /** The scores settled. Settling on [[EdgeClass.NoEdge]] is where the relationships of a record
    * stop, and a position holding none relates nothing.
    */
  def decide(logits: EdgeLogits[V]): RecordEdges[Edge] =
    val edgeClass = logits.edgeClass.argmax(Axis[EdgeClasses])
    val relates = !(edgeClass elementEquals_! EdgeClass.NoEdge.id)
    def named(scores: Tensor2[Edge, Node, V]) = where_!(relates, scores.argmax(Axis[Node]), 0)
    RecordEdges(edgeClass = edgeClass, subject = named(logits.subject), obj = named(logits.obj))

object EdgeHead:

  /** A drawing's relationships, scored: what each is, and a node for either end. */
  case class EdgeLogits[V](
      edgeClass: Tensor2[Edge, EdgeClasses, V],
      subject: Tensor2[Edge, Node, V],
      obj: Tensor2[Edge, Node, V]
  )

  case class Params[V](
      edgeClass: AffineLayer.Params[Embedding, EdgeClasses, V],
      subject: Pointer.Params[V],
      obj: Pointer.Params[V]
  )

  object Params:
    given tensorTree: TensorTree[Params[Float32]] = TensorTree.derived
    given tree: TreeOf[Params[Float32], Float32] = TreeOf.derived
