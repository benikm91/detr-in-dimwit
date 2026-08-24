import deepwit.attention.{MultiHeadFullSelfAttention, MultiHeadSelfAttention}
import deepwit.normalization.LayerNorm
import deepwit.transformer.TransformerBlock
import dimwit.*
import dimwit.Label as Λ

/** The encoder of DETR, [[https://arxiv.org/abs/2005.12872 End-to-End Object Detection with Transformers]],
  * in its pre-norm variant (`normalize_before` in the reference implementation).
  *
  * A stack of [[DETREncoderBlock]]s followed by a final normalization. Composed of deepwit's
  * building blocks rather than provided by them, so that every choice below can be changed here.
  *
  * @tparam Context The axis label for the sequence, i.e. the image patches.
  * @tparam Embedding The axis label for the embedding space.
  * @tparam V The floating-point scalar type of the tensor elements.
  * @param contextAxis The axis of the sequence attending onto itself.
  * @param params The learnable parameters.
  */
class DETREncoder[Context: Λ, Embedding: Λ, V: IsFloating](
    contextAxis: Axis[Context],
    params: DETREncoder.Params[Embedding, V]
) extends (Tensor2[Context, Embedding, V] => Tensor2[Context, Embedding, V]):

  private val blocks = params.encoderBlocks.map(p => DETREncoderBlock(contextAxis, p))
  private val finalNorm = LayerNorm(params.finalNorm)

  override def apply(context: Tensor2[Context, Embedding, V]): Tensor2[Context, Embedding, V] =
    val res = blocks.foldLeft(context):
      case (context_i, block) => block(context_i)
    res.vmap(Axis[Context])(finalNorm)

object DETREncoder:

  case class Params[Embedding, V](
      encoderBlocks: List[DETREncoderBlock.Params[Embedding, V]],
      finalNorm: LayerNorm.Params[Embedding, V]
  )

  object Params:

    def xavierUniformDepthScaled[Embedding: Λ, V: IsFloating](numEncoderBlocks: Int, numHeads: Int, embeddingExtent: AxisExtent[Embedding], embeddingMixedExtent: AxisExtent[EmbeddingMixed], key: Key, vtype: VType[V] = VType[Float32]): Params[Embedding, V] =
      new Params[Embedding, V](
        encoderBlocks =
          key.split(numEncoderBlocks).map: key =>
            DETREncoderBlock.Params.xavierUniformDepthScaled(numEncoderBlocks, numHeads, embeddingExtent, embeddingMixedExtent, key, vtype)
          .toList,
        finalNorm = LayerNorm.Params.identity(embeddingExtent, vtype)
      )

/** A single encoder block: the patches attend onto themselves, then along the embedding.
  *
  * Attention is unrestricted, as an image is not a sequence with a past — every patch may see every
  * other one. Normalization sits ahead of both residual branches, and the embedding mixer is a GELU
  * MLP.
  *
  * @tparam Context The axis label for the sequence, i.e. the image patches.
  * @tparam Embedding The axis label for the embedding space.
  * @tparam V The floating-point scalar type of the tensor elements.
  * @param contextAxis The axis of the sequence attending onto itself.
  * @param params The learnable parameters.
  */
class DETREncoderBlock[Context: Λ, Embedding: Λ, V: IsFloating](
    contextAxis: Axis[Context],
    params: DETREncoderBlock.Params[Embedding, V]
) extends TransformerBlock[Context, Embedding, V](contextAxis):

  private val selfAttention = MultiHeadFullSelfAttention(contextAxis, params.attentionParams)
  private val selfAttentionPreNorm = LayerNorm(params.attentionNormParams)

  private val mlp = MLPEmbeddingMixer(params.mlpParams)
  private val mlpPreNorm = LayerNorm(params.mlpNormParams)

  override protected def embeddingMixer(embedding: Tensor1[Embedding, V]): Tensor1[Embedding, V] =
    mlp(mlpPreNorm(embedding))

  override protected def contextMixer(context: Tensor2[Context, Embedding, V]): Tensor2[Context, Embedding, V] =
    selfAttention(context.vmap(Axis[Context])(selfAttentionPreNorm))

object DETREncoderBlock:

  case class Params[Embedding, V](
      attentionParams: MultiHeadSelfAttention.Params[Embedding, V],
      attentionNormParams: LayerNorm.Params[Embedding, V],
      mlpParams: MLPEmbeddingMixer.Params[Embedding, V],
      mlpNormParams: LayerNorm.Params[Embedding, V]
  )

  object Params:

    def xavierUniformDepthScaled[Embedding: Λ, V: IsFloating](numEncoderBlocks: Int, numHeads: Int, embeddingExtent: AxisExtent[Embedding], embeddingMixedExtent: AxisExtent[EmbeddingMixed], key: Key, vtype: VType[V] = VType[Float32]): Params[Embedding, V] =
      val (attnKey, mixKey) = key.splitToTuple(2)
      new Params[Embedding, V](
        attentionParams = MultiHeadSelfAttention.Params.xavierUniformDepthScaled(numEncoderBlocks, numHeads, embeddingExtent, attnKey, vtype),
        attentionNormParams = LayerNorm.Params.identity(embeddingExtent, vtype),
        mlpParams = MLPEmbeddingMixer.Params.xavierUniform(embeddingExtent, embeddingMixedExtent, mixKey, vtype),
        mlpNormParams = LayerNorm.Params.identity(embeddingExtent, vtype)
      )
