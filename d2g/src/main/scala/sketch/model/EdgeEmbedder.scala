package d2g.sketch.model

import d2g.*
import dataset.EdgeClasses
import dataset.RecordEdges
import deepwit.base.AffineLayer
import deepwit.embedder.VocabularyEmbedder
import dimwit.*

import scala.language.implicitConversions

/** Transforms a drawing's relationship into a single embedding vector: what it is, and the two
  * nodes it relates as `nodes` carries them.
  */
class EdgeEmbedder[V: IsFloating](params: EdgeEmbedder.Params[V]):

  private val edgeClass = VocabularyEmbedder(params.edgeClass)
  private val subject = AffineLayer(params.subject)
  private val obj = AffineLayer(params.obj)

  def apply(edges: RecordEdges[Edge], nodes: Tensor2[Node, Embedding, V]): Tensor2[Edge, Embedding, V] =
    val (subjects, objs) = (nodes.slice(Axis[Node].at(edges.subject)), nodes.slice(Axis[Node].at(edges.obj)))
    edges.edgeClass.vmap(Axis[Edge])(edgeClass) + subjects.vmap(Axis[Edge])(subject) + objs.vmap(Axis[Edge])(obj)

object EdgeEmbedder:

  case class Params[V](
      edgeClass: VocabularyEmbedder.Params[EdgeClasses, Embedding, V],
      subject: AffineLayer.Params[Embedding, Embedding, V],
      obj: AffineLayer.Params[Embedding, Embedding, V]
  )

  object Params:
    given tensorTree: TensorTree[Params[Float32]] = TensorTree.derived
    given tree: TreeOf[Params[Float32], Float32] = TreeOf.derived
