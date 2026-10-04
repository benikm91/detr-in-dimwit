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
import deepwit.embedder.VocabularyEmbedder
import dimwit.*

import scala.language.implicitConversions

trait EdgePart derives Label // The parts a edge embedding is composed of

/** Transforms a record edge into a single embedding vector. */
class EdgeEmbedder[V: IsFloating](params: EdgeEmbedder.Params[V]) extends (RecordEdges[Edge] => Tensor2[Edge, Embedding, V]):

  private val edgeClass = VocabularyEmbedder(params.edgeClass)
  private val subject = VocabularyEmbedder(params.subject)
  private val obj = VocabularyEmbedder(params.obj)
  private val project = AffineLayer(params.projection)

  override def apply(edges: RecordEdges[Edge]): Tensor2[Edge, Embedding, V] =
    zipvmap(Axis[Edge])(edges.edgeClass, edges.subject, edges.obj):
      case (cls, subj, ob) =>
        val parts = Seq(edgeClass(cls), subject(subj), obj(ob))
        project(stack(parts, Axis[EdgePart]).flatten((Axis[EdgePart], Axis[PartEmbedding])))

object EdgeEmbedder:

  case class Params[V](
      edgeClass: VocabularyEmbedder.Params[EdgeClasses, PartEmbedding, V],
      subject: VocabularyEmbedder.Params[LinkedNode, PartEmbedding, V],
      obj: VocabularyEmbedder.Params[LinkedNode, PartEmbedding, V],
      projection: AffineLayer.Params[EdgePart |*| PartEmbedding, Embedding, V]
  )

  object Params:
    given tensorTree: TensorTree[Params[Float32]] = TensorTree.derived
    given tree: TreeOf[Params[Float32], Float32] = TreeOf.derived
