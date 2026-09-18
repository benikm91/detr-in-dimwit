package documentEncoder

import deepwit.activation.gelu
import deepwit.attention.MultiHeadFullSelfAttention
import deepwit.attention.MultiHeadSelfAttention
import deepwit.base.AffineLayer
import deepwit.normalization.LayerNorm
import deepwit.transformer.TransformerBlock
import dimwit.*
import dimwit.Label as Λ

/** A vision transformer over the document: its patches embedded, then full self-attention among them.
  *
  * @param width The axis of the document width; the document must cut into whole patches.
  * @param height The axis of the document height; likewise.
  * @param channel The axis of the single drawing channel.
  */
class DocumentEncoder[Width: Λ, Height: Λ, Channel: Λ, Embedding: Λ, V: IsFloating](
    width: Axis[Width],
    height: Axis[Height],
    channel: Axis[Channel],
    params: DocumentEncoder.Params[Embedding, V]
) extends (Tensor3[Width, Height, Channel, V] => Tensor2[Width |*| Height, Embedding, V]):

  private val patches = ImageToPatchEmbedder(width, height, channel, params.patchEmbedder)
  private val blocks = params.blocks.map(DocumentEncoderBlock(Axis[Width |*| Height], _))
  private val finalNorm = LayerNorm(params.finalNorm)

  override def apply(document: Tensor3[Width, Height, Channel, V]): Tensor2[Width |*| Height, Embedding, V] =
    blocks
      .foldLeft(patches(document))((encoded, block) => block(encoded))
      .vmap(Axis[Width |*| Height])(finalNorm)

object DocumentEncoder:

  /** The widened space an embedding is mixed in between attentions. */
  trait EmbeddingMixed derives Label

  case class Params[Embedding, V](
      patchEmbedder: ImageToPatchEmbedder.Params[Embedding, V],
      blocks: List[DocumentEncoderBlock.Params[Embedding, V]],
      finalNorm: LayerNorm.Params[Embedding, V]
  )

  object Params:

    def xavierUniformDepthScaled[Embedding: Λ, V: IsFloating](numBlocks: Int, numHeads: Int, embeddingExtent: AxisExtent[Embedding], embeddingMixedExtent: AxisExtent[DocumentEncoder.EmbeddingMixed], key: Key, vtype: VType[V] = VType[Float32]): Params[Embedding, V] =
      val (patchKey, blocksKey) = key.splitToTuple(2)
      Params(
        patchEmbedder = ImageToPatchEmbedder.Params.xavierUniform(embeddingExtent, patchKey, vtype),
        blocks = blocksKey.split(numBlocks).map(DocumentEncoderBlock.Params.xavierUniformDepthScaled(numBlocks, numHeads, embeddingExtent, embeddingMixedExtent, _, vtype)).toList,
        finalNorm = LayerNorm.Params.identity(embeddingExtent, vtype)
      )

/** Single [[TransformerBlock]] in [[DocumentEncoder]]: the patches attend onto themselves, then along the embedding. */
class DocumentEncoderBlock[Patch: Λ, Embedding: Λ, V: IsFloating](patch: Axis[Patch], params: DocumentEncoderBlock.Params[Embedding, V]) extends TransformerBlock[Patch, Embedding, V](patch):

  private val selfAttention = MultiHeadFullSelfAttention(patch, params.selfAttention)
  private val selfAttentionPreNorm = LayerNorm(params.selfAttentionNorm)
  private val mlp = MLPEmbeddingMixer(params.mlp)
  private val mlpPreNorm = LayerNorm(params.mlpNorm)

  override protected def contextMixer(patches: Tensor2[Patch, Embedding, V]): Tensor2[Patch, Embedding, V] =
    selfAttention(patches.vmap(patch)(selfAttentionPreNorm))

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

    def xavierUniformDepthScaled[Embedding: Λ, V: IsFloating](numBlocks: Int, numHeads: Int, embeddingExtent: AxisExtent[Embedding], embeddingMixedExtent: AxisExtent[DocumentEncoder.EmbeddingMixed], key: Key, vtype: VType[V] = VType[Float32]): Params[Embedding, V] =
      val (attentionKey, mlpKey) = key.splitToTuple(2)
      Params(
        selfAttention = MultiHeadSelfAttention.Params.xavierUniformDepthScaled(numBlocks, numHeads, embeddingExtent, attentionKey, vtype),
        selfAttentionNorm = LayerNorm.Params.identity(embeddingExtent, vtype),
        mlp = MLPEmbeddingMixer.Params.xavierUniform(embeddingExtent, embeddingMixedExtent, mlpKey, vtype),
        mlpNorm = LayerNorm.Params.identity(embeddingExtent, vtype)
      )

/** MLP embedding mixer inside a [[DocumentEncoderBlock]]. */
class MLPEmbeddingMixer[Embedding: Λ, V: IsFloating](params: MLPEmbeddingMixer.Params[Embedding, V]) extends (Tensor1[Embedding, V] => Tensor1[Embedding, V]):

  private val expand = AffineLayer(params.expand)
  private val project = AffineLayer(params.project)

  override def apply(embedding: Tensor1[Embedding, V]): Tensor1[Embedding, V] = project(gelu(expand(embedding)))

object MLPEmbeddingMixer:

  case class Params[Embedding, V](
      expand: AffineLayer.Params[Embedding, DocumentEncoder.EmbeddingMixed, V],
      project: AffineLayer.Params[DocumentEncoder.EmbeddingMixed, Embedding, V]
  )

  object Params:

    def xavierUniform[Embedding: Λ, V: IsFloating](embeddingExtent: AxisExtent[Embedding], embeddingMixedExtent: AxisExtent[DocumentEncoder.EmbeddingMixed], key: Key, vtype: VType[V] = VType[Float32]): Params[Embedding, V] =
      val (expandKey, projectKey) = key.splitToTuple(2)
      Params(
        expand = AffineLayer.Params.xavierUniform(embeddingExtent, embeddingMixedExtent, expandKey, vtype),
        project = AffineLayer.Params.xavierUniform(embeddingMixedExtent, embeddingExtent, projectKey, vtype)
      )
