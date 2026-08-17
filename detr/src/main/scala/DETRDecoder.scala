import deepwit.attention.{MultiHeadAttention, MultiHeadFullAttention, MultiHeadFullSelfAttention, MultiHeadSelfAttention}
import deepwit.normalization.LayerNorm
import deepwit.transformer.CrossTransformerBlock
import dimwit.*
import dimwit.Label as Λ

/** The decoder of DETR, [[https://arxiv.org/abs/2005.12872 End-to-End Object Detection with Transformers]],
  * in its pre-norm variant (`normalize_before` in the reference implementation).
  *
  * A stack of [[DETRDecoderBlock]]s followed by a final normalization. Composed of deepwit's
  * building blocks rather than provided by them, so that every choice below can be changed here.
  *
  * @tparam CrossContext The axis label for the cross sequence, i.e. the encoded image patches.
  * @tparam CrossEmbedding The axis label for the cross embedding space.
  * @tparam Context The axis label for the sequence, i.e. the object queries.
  * @tparam Embedding The axis label for the embedding space.
  * @tparam V The floating-point scalar type of the tensor elements.
  * @param crossContextAxis The axis of the sequence being attended onto.
  * @param contextAxis The axis of the sequence attending onto itself and onto the cross context.
  * @param params The learnable parameters.
  */
class DETRDecoder[CrossContext: Λ, CrossEmbedding: Λ, Context: Λ, Embedding: Λ, V: IsFloating](
    crossContextAxis: Axis[CrossContext],
    contextAxis: Axis[Context],
    params: DETRDecoder.Params[CrossEmbedding, Embedding, V]
) extends ((Tensor2[CrossContext, CrossEmbedding, V], Tensor2[Context, Embedding, V]) => Tensor2[Context, Embedding, V]):

  private val blocks = params.decoderBlocks.map(p => DETRDecoderBlock(crossContextAxis, contextAxis, p))
  private val finalNorm = LayerNorm(params.finalNorm)

  override def apply(crossContext: Tensor2[CrossContext, CrossEmbedding, V], context: Tensor2[Context, Embedding, V]): Tensor2[Context, Embedding, V] =
    val res = blocks.foldLeft(context):
      case (context_i, block) => block(crossContext, context_i)
    res.vmap(Axis[Context])(finalNorm)

  /** The same forward pass, keeping every block's output for the auxiliary decoding losses. */
  def applyWithHiddenStates(crossContext: Tensor2[CrossContext, CrossEmbedding, V], context: Tensor2[Context, Embedding, V]): (List[Tensor2[Context, Embedding, V]], Tensor2[Context, Embedding, V]) =
    val allStates = blocks.scanLeft(context):
      case (context_i, block) => block(crossContext, context_i)
    val hiddenStates = allStates.tail // drop initial context
    val res = hiddenStates.last
    (hiddenStates, res.vmap(Axis[Context])(finalNorm))

  def applyWithSelfAttentionIntermediates(crossContext: Tensor2[CrossContext, CrossEmbedding, V], context: Tensor2[Context, Embedding, V]): (List[MultiHeadAttention.Intermediates[Context, Context, V]], Tensor2[Context, Embedding, V]) =
    val (blocksRes, intermediates) = blocks.foldLeft((context, List.empty[MultiHeadAttention.Intermediates[Context, Context, V]])):
      case ((context_i, sofar), block) =>
        val (mixed, intermediates) = block.applyWithSelfAttentionIntermediates(crossContext, context_i)
        (mixed, intermediates :: sofar)
    val res = blocksRes.vmap(Axis[Context])(finalNorm)
    (intermediates.reverse, res)

object DETRDecoder:

  case class Params[CrossEmbedding, Embedding, V](
      decoderBlocks: List[DETRDecoderBlock.Params[CrossEmbedding, Embedding, V]],
      finalNorm: LayerNorm.Params[Embedding, V]
  )

  object Params:

    def xavierUniformDepthScaled[CrossEmbedding: Λ, Embedding: Λ, V: IsFloating](numDecoderBlocks: Int, numHeads: Int, crossEmbeddingExtent: AxisExtent[CrossEmbedding], embeddingExtent: AxisExtent[Embedding], embeddingMixedExtent: AxisExtent[EmbeddingMixed], vtype: VType[V], key: Key): Params[CrossEmbedding, Embedding, V] =
      new Params(
        decoderBlocks =
          key.split(numDecoderBlocks).map: key =>
            DETRDecoderBlock.Params.xavierUniformDepthScaled(numDecoderBlocks, numHeads, crossEmbeddingExtent, embeddingExtent, embeddingMixedExtent, vtype, key)
          .toList,
        finalNorm = LayerNorm.Params.identity(embeddingExtent, vtype)
      )

/** A single decoder block: the queries attend onto themselves, then onto the encoded image, then
  * along the embedding.
  *
  * Attention is unrestricted in both directions, as the object queries are a set rather than a
  * sequence — every query has to see every other one to settle what it stands for. Normalization
  * sits ahead of all three residual branches, and the embedding mixer is a GELU MLP.
  *
  * @tparam CrossContext The axis label for the cross sequence, i.e. the encoded image patches.
  * @tparam CrossEmbedding The axis label for the cross embedding space.
  * @tparam Context The axis label for the sequence, i.e. the object queries.
  * @tparam Embedding The axis label for the embedding space.
  * @tparam V The floating-point scalar type of the tensor elements.
  * @param crossContextAxis The axis of the sequence being attended onto.
  * @param contextAxis The axis of the sequence attending onto itself and onto the cross context.
  * @param params The learnable parameters.
  */
class DETRDecoderBlock[CrossContext: Λ, CrossEmbedding: Λ, Context: Λ, Embedding: Λ, V: IsFloating](
    crossContextAxis: Axis[CrossContext],
    contextAxis: Axis[Context],
    params: DETRDecoderBlock.Params[CrossEmbedding, Embedding, V]
) extends CrossTransformerBlock[CrossContext, CrossEmbedding, Context, Embedding, V](crossContextAxis, contextAxis):

  private val selfAttention = MultiHeadFullSelfAttention[Context, Embedding, V](contextAxis, params.selfAttentionParams)
  private val selfAttentionPreNorm = LayerNorm(params.selfAttentionNormParams)

  private val crossAttention = MultiHeadFullAttention(crossContextAxis, contextAxis, params.crossAttentionParams)
  private val crossAttentionPreNorm = LayerNorm(params.crossAttentionNormParams)

  private val mlp = MLPEmbeddingMixer(params.mlpParams)
  private val mlpPreNorm = LayerNorm(params.mlpNormParams)

  override protected def embeddingMixer(embedding: Tensor1[Embedding, V]): Tensor1[Embedding, V] =
    mlp(mlpPreNorm(embedding))

  override protected def contextMixer(context: Tensor2[Context, Embedding, V]): Tensor2[Context, Embedding, V] =
    selfAttend(context).head

  override protected def crossContextMixer(crossContext: Tensor2[CrossContext, CrossEmbedding, V], context: Tensor2[Context, Embedding, V]): Tensor2[Context, Embedding, V] =
    crossAttention(crossContext, context.vmap(Axis[Context])(crossAttentionPreNorm))

  def applyWithSelfAttentionIntermediates(crossContext: Tensor2[CrossContext, CrossEmbedding, V], context: Tensor2[Context, Embedding, V]): (
      Tensor2[Context, Embedding, V],
      MultiHeadAttention.Intermediates[Context, Context, V]
  ) =
    val (selfAttended, intermediates) = selfAttend(context)
    val contextMixed = context + selfAttended
    val crossContextMixed = contextMixed + crossContextMixer(crossContext, contextMixed)
    val res = crossContextMixed + crossContextMixed.vmap(Axis[Context])(embeddingMixer)
    (res, intermediates)

  /** The self-attention branch, together with the queries, keys and values it attended by. */
  private def selfAttend(context: Tensor2[Context, Embedding, V]): (
      Tensor2[Context, Embedding, V],
      MultiHeadAttention.Intermediates[Context, Context, V]
  ) =
    selfAttention.applyWithIntermediates(context.vmap(Axis[Context])(selfAttentionPreNorm))

object DETRDecoderBlock:

  case class Params[CrossEmbedding, Embedding, V](
      crossAttentionParams: MultiHeadAttention.Params[CrossEmbedding, Embedding, V],
      crossAttentionNormParams: LayerNorm.Params[Embedding, V],
      selfAttentionParams: MultiHeadSelfAttention.Params[Embedding, V],
      selfAttentionNormParams: LayerNorm.Params[Embedding, V],
      mlpNormParams: LayerNorm.Params[Embedding, V],
      mlpParams: MLPEmbeddingMixer.Params[Embedding, V]
  )

  object Params:

    def xavierUniformDepthScaled[CrossEmbedding: Λ, Embedding: Λ, V: IsFloating](numDecoderBlocks: Int, numHeads: Int, crossEmbeddingExtent: AxisExtent[CrossEmbedding], embeddingExtent: AxisExtent[Embedding], embeddingMixedExtent: AxisExtent[EmbeddingMixed], vtype: VType[V], key: Key): Params[CrossEmbedding, Embedding, V] =
      val (selfAttnKey, crossAttnKey, mixKey) = key.splitToTuple(3)
      new Params[CrossEmbedding, Embedding, V](
        crossAttentionParams = MultiHeadAttention.Params.xavierUniformDepthScaled(numDecoderBlocks, numHeads, crossEmbeddingExtent, embeddingExtent, vtype, crossAttnKey),
        crossAttentionNormParams = LayerNorm.Params.identity(embeddingExtent, vtype),
        selfAttentionParams = MultiHeadSelfAttention.Params.xavierUniformDepthScaled(numDecoderBlocks, numHeads, embeddingExtent, vtype, selfAttnKey),
        selfAttentionNormParams = LayerNorm.Params.identity(embeddingExtent, vtype),
        mlpParams = MLPEmbeddingMixer.Params.xavierUniform(embeddingExtent, embeddingMixedExtent, vtype, mixKey),
        mlpNormParams = LayerNorm.Params.identity(embeddingExtent, vtype)
      )
