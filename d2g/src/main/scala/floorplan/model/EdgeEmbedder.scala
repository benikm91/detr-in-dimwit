package d2g.floorplan.model

import d2g.*
import d2s.floorplan.model.NodeEmbedder.PartEmbedding
import deepwit.base.AffineLayer
import deepwit.embedder.VocabularyEmbedder
import dimwit.*
import syntheticFloorplansDataset.FloorplanEdges
import syntheticFloorplansDataset.FloorplanRelationships

import scala.language.implicitConversions

import EdgeEmbedder.EdgePart

/** Transforms a floor plan's relationship — what it is, and the two nodes it relates — into a single
  * embedding vector.
  */
class EdgeEmbedder[V: IsFloating](params: EdgeEmbedder.Params[V]) extends (FloorplanEdges[Edge] => Tensor2[Edge, Embedding, V]):

  private val relationship = VocabularyEmbedder(params.relationship)
  private val subject = VocabularyEmbedder(params.subject)
  private val obj = VocabularyEmbedder(params.obj)
  private val project = AffineLayer(params.projection)

  override def apply(edges: FloorplanEdges[Edge]): Tensor2[Edge, Embedding, V] =
    zipvmap(Axis[Edge])(edges.relationship, edges.subject, edges.obj):
      case (related, subj, ob) =>
        val parts = Seq(relationship(related), subject(subj), obj(ob))
        project(stack(parts, Axis[EdgePart]).flatten((Axis[EdgePart], Axis[PartEmbedding])))

object EdgeEmbedder:

  trait EdgePart derives Label // The parts an edge embedding is composed of

  case class Params[V](
      relationship: VocabularyEmbedder.Params[FloorplanRelationships, PartEmbedding, V],
      subject: VocabularyEmbedder.Params[LinkedNode, PartEmbedding, V],
      obj: VocabularyEmbedder.Params[LinkedNode, PartEmbedding, V],
      projection: AffineLayer.Params[EdgePart |*| PartEmbedding, Embedding, V]
  )

  object Params:
    given tensorTree: TensorTree[Params[Float32]] = TensorTree.derived
    given tree: TreeOf[Params[Float32], Float32] = TreeOf.derived
