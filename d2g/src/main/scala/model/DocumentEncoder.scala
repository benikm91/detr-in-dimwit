package d2g.model

import d2g.*
import d2g.train.*
import d2g.eval.*
import d2g.config.*
import deepwit.activation.gelu
import deepwit.attention.AttentionScore
import deepwit.attention.MultiHeadAttention
import deepwit.attention.MultiHeadCustomAttention
import deepwit.attention.MultiHeadCustomSelfAttention
import deepwit.attention.MultiHeadFullAttention
import deepwit.attention.MultiHeadFullSelfAttention
import deepwit.attention.MultiHeadSelfAttention
import deepwit.base.AffineLayer
import deepwit.normalization.LayerNorm
import deepwit.transformer.TransformerBlock
import dimwit.*
import dimwit.Conversions.given
import dimwit.Label as Λ

import scala.language.implicitConversions

/** A full self-attention encoder of the document's patches. */
class DocumentEncoder[Embedding: Λ, V: IsFloating](params: DocumentEncoder.Params[Embedding, V]) extends (Tensor2[Patch, Embedding, V] => Tensor2[Patch, Embedding, V]):

  private val blocks = params.blocks.map(DocumentEncoderBlock(_))
  private val finalNorm = LayerNorm(params.finalNorm)

  override def apply(patches: Tensor2[Patch, Embedding, V]): Tensor2[Patch, Embedding, V] =
    blocks
      .foldLeft(patches)((encoded, block) => block(encoded))
      .vmap(Axis[Patch])(finalNorm)

object DocumentEncoder:

  case class Params[Embedding, V](
      blocks: List[DocumentEncoderBlock.Params[Embedding, V]],
      finalNorm: LayerNorm.Params[Embedding, V]
  )

  object Params:

    def xavierUniformDepthScaled[Embedding: Λ, V: IsFloating](numBlocks: Int, numHeads: Int, embeddingExtent: AxisExtent[Embedding], embeddingMixedExtent: AxisExtent[EmbeddingMixed], key: Key, vtype: VType[V] = VType[Float32]): Params[Embedding, V] =
      Params(
        blocks = key.split(numBlocks).map(DocumentEncoderBlock.Params.xavierUniformDepthScaled(numBlocks, numHeads, embeddingExtent, embeddingMixedExtent, _, vtype)).toList,
        finalNorm = LayerNorm.Params.identity(embeddingExtent, vtype)
      )

/** Single [[TransformerBlock]] in [[DocumentEncoder]] */
class DocumentEncoderBlock[Embedding: Λ, V: IsFloating](params: DocumentEncoderBlock.Params[Embedding, V]) extends TransformerBlock[Patch, Embedding, V](Axis[Patch]):

  private val selfAttention = MultiHeadFullSelfAttention(Axis[Patch], params.selfAttention)
  private val selfAttentionPreNorm = LayerNorm(params.selfAttentionNorm)
  private val mlp = MLPEmbeddingMixer(params.mlp)
  private val mlpPreNorm = LayerNorm(params.mlpNorm)

  override protected def contextMixer(patches: Tensor2[Patch, Embedding, V]): Tensor2[Patch, Embedding, V] =
    selfAttention(patches.vmap(Axis[Patch])(selfAttentionPreNorm))

  override protected def embeddingMixer(embedding: Tensor1[Embedding, V]): Tensor1[Embedding, V] =
    mlp(mlpPreNorm(embedding))

object DocumentEncoderBlock:

  case class Params[Embedding, V](
      selfAttention: MultiHeadSelfAttention.Params[Embedding, V],
      selfAttentionNorm: LayerNorm.Params[Embedding, V],
      mlp: MLPEmbeddingMixer.Params[Embedding, V],
      mlpNorm: LayerNorm.Params[Embedding, V]
  )

  object Params:

    def xavierUniformDepthScaled[Embedding: Λ, V: IsFloating](numBlocks: Int, numHeads: Int, embeddingExtent: AxisExtent[Embedding], embeddingMixedExtent: AxisExtent[EmbeddingMixed], key: Key, vtype: VType[V] = VType[Float32]): Params[Embedding, V] =
      val (attentionKey, mlpKey) = key.splitToTuple(2)
      Params(
        selfAttention = MultiHeadSelfAttention.Params.xavierUniformDepthScaled(numBlocks, numHeads, embeddingExtent, attentionKey, vtype),
        selfAttentionNorm = LayerNorm.Params.identity(embeddingExtent, vtype),
        mlp = MLPEmbeddingMixer.Params.init(embeddingExtent, embeddingMixedExtent, mlpKey, vtype),
        mlpNorm = LayerNorm.Params.identity(embeddingExtent, vtype)
      )
