package d2g.floorplan.model

import d2g.*
import d2g.model.*
import deepwit.base.AffineLayer
import dimwit.*
import syntheticFloorplansDataset.FloorplanEdges
import syntheticFloorplansDataset.FloorplanRelationship
import syntheticFloorplansDataset.FloorplanRelationships

import scala.language.implicitConversions

/** Transforms an embedding back into a floor plan's relationship (logits): what it is, and a
  * [[Pointer]] at each of the two nodes it relates.
  */
class EdgeHead[V: IsFloating](params: EdgeHead.Params[V]):

  import EdgeHead.EdgeLogits

  private val relationship = AffineLayer(params.relationship)
  private val subject = Pointer(params.subject)
  private val obj = Pointer(params.obj)

  def apply(embeddings: Tensor2[Edge, Embedding, V], nodes: NodeSource[Embedding, V]): EdgeLogits[V] =
    EdgeLogits(
      relationship = embeddings.vmap(Axis[Edge])(relationship),
      subject = subject(embeddings, nodes),
      obj = obj(embeddings, nodes)
    )

  /** The scores settled. Settling on [[FloorplanRelationship.NoRelationship]] is where the
    * relationships of a record stop, and a position holding none relates nothing.
    */
  def decide(logits: EdgeLogits[V]): FloorplanEdges[Edge] =
    val relationship = logits.relationship.argmax(Axis[FloorplanRelationships])
    val relates = !(relationship elementEquals_! FloorplanRelationship.NoRelationship.id)
    def named(scores: Tensor2[Edge, Node, V]) =
      val end = scores.argmax(Axis[Node])
      where_!(relates, end, 0)
    FloorplanEdges(relationship = relationship, subject = named(logits.subject), obj = named(logits.obj))

object EdgeHead:

  /** A floor plan's relationships, scored: what each is, and a node for either end. */
  case class EdgeLogits[V](
      relationship: Tensor2[Edge, FloorplanRelationships, V],
      subject: Tensor2[Edge, Node, V],
      obj: Tensor2[Edge, Node, V]
  )

  case class Params[V](
      relationship: AffineLayer.Params[Embedding, FloorplanRelationships, V],
      subject: Pointer.Params[V],
      obj: Pointer.Params[V]
  )

  object Params:
    given tensorTree: TensorTree[Params[Float32]] = TensorTree.derived
    given tree: TreeOf[Params[Float32], Float32] = TreeOf.derived
