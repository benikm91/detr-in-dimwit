package d2g.floorplan.model

import d2g.*
import deepwit.base.AffineLayer
import deepwit.embedder.VocabularyEmbedder
import dimwit.*
import syntheticFloorplansDataset.FloorplanEdges
import syntheticFloorplansDataset.FloorplanRelationships

import scala.language.implicitConversions

/** Transforms a floor plan's relationship into a single embedding vector: what it is, and the two
  * nodes it relates as `nodes` carries them.
  */
class EdgeEmbedder[V: IsFloating](params: EdgeEmbedder.Params[V]):

  private val relationship = VocabularyEmbedder(params.relationship)
  private val subject = AffineLayer(params.subject)
  private val obj = AffineLayer(params.obj)

  def apply(edges: FloorplanEdges[Edge], nodes: Tensor2[Node, Embedding, V]): Tensor2[Edge, Embedding, V] =
    val (subjects, objs) = (nodes.slice(Axis[Node].at(edges.subject)), nodes.slice(Axis[Node].at(edges.obj)))
    edges.relationship.vmap(Axis[Edge])(relationship) + subjects.vmap(Axis[Edge])(subject) + objs.vmap(Axis[Edge])(obj)

object EdgeEmbedder:

  case class Params[V](
      relationship: VocabularyEmbedder.Params[FloorplanRelationships, Embedding, V],
      subject: AffineLayer.Params[Embedding, Embedding, V],
      obj: AffineLayer.Params[Embedding, Embedding, V]
  )

  object Params:
    given tensorTree: TensorTree[Params[Float32]] = TensorTree.derived
    given tree: TreeOf[Params[Float32], Float32] = TreeOf.derived
