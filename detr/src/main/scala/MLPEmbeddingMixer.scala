import dimwit.*
import deepwit.activation.gelu
import deepwit.base.AffineLayer
import dimwit.Label as Λ

/** The axis label for the widened space an embedding is mixed in by an [[MLPEmbeddingMixer]]. */
trait EmbeddingMixed derives Label

/** Mixes the components of a single embedding through a two-layer MLP.
  *
  * The embedding is expanded into the wider [[EmbeddingMixed]] space, passed through the
  * activation function, and projected back into the embedding space.
  *
  * @tparam Embedding The axis label for the embedding space.
  * @tparam V The floating-point scalar type of the tensor elements.
  * @param params The learnable parameters.
  * @param activation The activation function applied to the expanded embedding.
  */
class MLPEmbeddingMixer[Embedding: Λ, V: IsFloating](
    params: MLPEmbeddingMixer.Params[Embedding, V]
) extends (Tensor1[Embedding, V] => Tensor1[Embedding, V]):

  private val expandLayer = AffineLayer(params.expand)
  private val projectLayer = AffineLayer(params.project)

  override def apply(embedding: Tensor1[Embedding, V]): Tensor1[Embedding, V] =
    val mixed = gelu(expandLayer(embedding))
    projectLayer(mixed)

object MLPEmbeddingMixer:

  case class Params[Embedding, V](
      expand: AffineLayer.Params[Embedding, EmbeddingMixed, V],
      project: AffineLayer.Params[EmbeddingMixed, Embedding, V]
  )

  object Params:

    def xavierUniform[Embedding: Λ, V: IsFloating](embeddingExtent: AxisExtent[Embedding], embeddingMixedExtent: AxisExtent[EmbeddingMixed], vtype: VType[V], key: Key): Params[Embedding, V] =
      val (expandKey, projectKey) = key.splitToTuple(2)
      Params(
        expand = AffineLayer.Params.xavierUniform(embeddingExtent, embeddingMixedExtent, vtype, expandKey),
        project = AffineLayer.Params.xavierUniform(embeddingMixedExtent, embeddingExtent, vtype, projectKey)
      )
