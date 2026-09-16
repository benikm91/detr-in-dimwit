package d2g.model

import dimwit.*
import dimwit.Label as Λ
import deepwit.base.AffineLayer
import deepwit.activation.gelu

trait EmbeddingMixed derives Label // Label for the widened vector space that mixes the embedding components

/** MLP embedding mixer inside the transformer block. */
class MLPEmbeddingMixer[Embedding: Λ, V: IsFloating](params: MLPEmbeddingMixer.Params[Embedding, V]) extends (Tensor1[Embedding, V] => Tensor1[Embedding, V]):

  private val expand = AffineLayer(params.expand)
  private val project = AffineLayer(params.project)

  override def apply(embedding: Tensor1[Embedding, V]): Tensor1[Embedding, V] = project(gelu(expand(embedding)))

object MLPEmbeddingMixer:

  case class Params[Embedding, V](
      expand: AffineLayer.Params[Embedding, EmbeddingMixed, V],
      project: AffineLayer.Params[EmbeddingMixed, Embedding, V]
  )

  object Params:

    def init[Embedding: Λ, V: IsFloating](embeddingExtent: AxisExtent[Embedding], embeddingMixedExtent: AxisExtent[EmbeddingMixed], key: Key, vtype: VType[V] = VType[Float32]): Params[Embedding, V] =
      val (expandKey, projectKey) = key.splitToTuple(2)
      Params(
        expand = AffineLayer.Params.init(embeddingExtent, embeddingMixedExtent, expandKey, vtype),
        project = AffineLayer.Params.init(embeddingMixedExtent, embeddingExtent, projectKey, vtype)
      )
