package d2g.model

import d2g.*
import deepwit.base.LinearLayer
import dimwit.*
import dimwit.Label as Λ

import scala.language.implicitConversions

/** Points at the node a relationship names, as the decoder of
  * [[https://arxiv.org/abs/1803.08475 Attention, Learn to Solve Routing Problems!]] (section 3.2)
  * points at the next node of a tour: a single head of scaled dot-product attention whose scores
  * are clipped to ±[[Pointer.Clip]] by a tanh, the highest score the node named. The nodes there
  * are not are never named.
  */
class Pointer[V: IsFloating](params: Pointer.Params[V]):

  private val query = LinearLayer(params.query)
  private val key = LinearLayer(params.key)

  def apply[Row: Λ](asking: Tensor2[Row, Embedding, V], nodes: NodeSource[Embedding, V]): Tensor2[Row, Node, V] =
    val queries = asking.vmap(Axis[Row])(query)
    val keys = nodes.nodeEmbeddings.vmap(Axis[Node])(key)
    val scaled = queries.dot(Axis[Embedding])(keys) /! math.sqrt(keys.shape(Axis[Embedding])).toFloat
    (scaled.tanh *! Pointer.Clip).vmap(Axis[Row])(scores => where(nodes.presentMask, scores, Tensor.like(scores).fill(Pointer.Absent)))

object Pointer:

  /** How far the scores reach either way. */
  val Clip = 10f

  /** The score of a node that is not there: far below any score a node that is there can get, and
    * finite, so that a plan without nodes still has scores to normalize.
    */
  private val Absent = -1e4f

  case class Params[V](query: LinearLayer.Params[Embedding, Embedding, V], key: LinearLayer.Params[Embedding, Embedding, V])

  object Params:

    def init(embedding: AxisExtent[Embedding], key: Key): Params[Float32] =
      val (queryKey, keyKey) = key.splitToTuple(2)
      Params(LinearLayer.Params.init(embedding, embedding, queryKey), LinearLayer.Params.init(embedding, embedding, keyKey))
