import deepwit.activation.gelu
import deepwit.attention.AttentionScore
import deepwit.attention.MultiHeadAttention
import deepwit.attention.MultiHeadCustomSelfAttention
import deepwit.attention.MultiHeadFullAttention
import deepwit.attention.MultiHeadFullSelfAttention
import deepwit.attention.MultiHeadSelfAttention
import deepwit.base.AffineLayer
import deepwit.normalization.LayerNorm
import deepwit.transformer.TransformerBlock
import dimwit.*
import dimwit.Label as Λ

/** The axis label for the widened space an embedding is mixed in by an [[MLPEmbeddingMixer]]. */
trait EmbeddingMixed derives Label

/** Mixes the components of a single embedding through a two-layer GELU MLP. */
class MLPEmbeddingMixer[Embedding: Λ, V: IsFloating](
    params: MLPEmbeddingMixer.Params[Embedding, V]
) extends (Tensor1[Embedding, V] => Tensor1[Embedding, V]):

  private val expand = AffineLayer(params.expand)
  private val project = AffineLayer(params.project)

  override def apply(embedding: Tensor1[Embedding, V]): Tensor1[Embedding, V] =
    project(gelu(expand(embedding)))

object MLPEmbeddingMixer:

  case class Params[Embedding, V](
      expand: AffineLayer.Params[Embedding, EmbeddingMixed, V],
      project: AffineLayer.Params[EmbeddingMixed, Embedding, V]
  )

  object Params:

    def xavierUniform[Embedding: Λ, V: IsFloating](embeddingExtent: AxisExtent[Embedding], embeddingMixedExtent: AxisExtent[EmbeddingMixed], key: Key, vtype: VType[V] = VType[Float32]): Params[Embedding, V] =
      val (expandKey, projectKey) = key.splitToTuple(2)
      Params(
        expand = AffineLayer.Params.xavierUniform(embeddingExtent, embeddingMixedExtent, expandKey, vtype),
        project = AffineLayer.Params.xavierUniform(embeddingMixedExtent, embeddingExtent, projectKey, vtype)
      )

/** The encoder of the document: a pre-norm transformer stack over the image patches.
  *
  * A drawing is not a sequence with a past, so every patch may see every other one.
  */
class DocumentEncoder[Patch: Λ, Embedding: Λ, V: IsFloating](
    patchAxis: Axis[Patch],
    params: DocumentEncoder.Params[Embedding, V]
) extends (Tensor2[Patch, Embedding, V] => Tensor2[Patch, Embedding, V]):

  private val blocks = params.blocks.map(block => DocumentEncoderBlock(patchAxis, block))
  private val finalNorm = LayerNorm(params.finalNorm)

  override def apply(patches: Tensor2[Patch, Embedding, V]): Tensor2[Patch, Embedding, V] =
    blocks.foldLeft(patches)((encoded, block) => block(encoded)).vmap(Axis[Patch])(finalNorm)

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

class DocumentEncoderBlock[Patch: Λ, Embedding: Λ, V: IsFloating](
    patchAxis: Axis[Patch],
    params: DocumentEncoderBlock.Params[Embedding, V]
) extends TransformerBlock[Patch, Embedding, V](patchAxis):

  private val selfAttention = MultiHeadFullSelfAttention(patchAxis, params.selfAttention)
  private val selfAttentionPreNorm = LayerNorm(params.selfAttentionNorm)
  private val mlp = MLPEmbeddingMixer(params.mlp)
  private val mlpPreNorm = LayerNorm(params.mlpNorm)

  override protected def embeddingMixer(embedding: Tensor1[Embedding, V]): Tensor1[Embedding, V] =
    mlp(mlpPreNorm(embedding))

  override protected def contextMixer(patches: Tensor2[Patch, Embedding, V]): Tensor2[Patch, Embedding, V] =
    selfAttention(patches.vmap(Axis[Patch])(selfAttentionPreNorm))

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
        mlp = MLPEmbeddingMixer.Params.xavierUniform(embeddingExtent, embeddingMixedExtent, mlpKey, vtype),
        mlpNorm = LayerNorm.Params.identity(embeddingExtent, vtype)
      )

/** The decoder of the record: a pre-norm transformer stack over the decoder sequence.
  *
  * The same block a detection transformer decodes with — masked self-attention, then
  * cross-attention onto the encoded document, then the embedding mixer, each on its own
  * residual branch — differing only in that the self-attention mask is handed in per
  * forward pass, since it depends on how much of the record has been taken.
  */
class RecordDecoder[Patch: Λ, PatchEmbedding: Λ, Slot: Λ, Embedding: Λ, V: IsFloating](
    patchAxis: Axis[Patch],
    slotAxis: Axis[Slot],
    params: RecordDecoder.Params[PatchEmbedding, Embedding, V]
):

  private val blocks = params.blocks.map(block => RecordDecoderBlock(patchAxis, slotAxis, block))
  private val finalNorm = LayerNorm(params.finalNorm)

  def apply(
      document: Tensor2[Patch, PatchEmbedding, V],
      sequence: Tensor2[Slot, Embedding, V],
      mask: Tensor2[Slot, Slot, Bool]
  ): Tensor2[Slot, Embedding, V] =
    blocks.foldLeft(sequence)((decoded, block) => block(document, decoded, mask)).vmap(Axis[Slot])(finalNorm)

object RecordDecoder:

  case class Params[PatchEmbedding, Embedding, V](
      blocks: List[RecordDecoderBlock.Params[PatchEmbedding, Embedding, V]],
      finalNorm: LayerNorm.Params[Embedding, V]
  )

  object Params:

    def xavierUniformDepthScaled[PatchEmbedding: Λ, Embedding: Λ, V: IsFloating](numBlocks: Int, numHeads: Int, patchEmbeddingExtent: AxisExtent[PatchEmbedding], embeddingExtent: AxisExtent[Embedding], embeddingMixedExtent: AxisExtent[EmbeddingMixed], key: Key, vtype: VType[V] = VType[Float32]): Params[PatchEmbedding, Embedding, V] =
      Params(
        blocks = key.split(numBlocks).map(RecordDecoderBlock.Params.xavierUniformDepthScaled(numBlocks, numHeads, patchEmbeddingExtent, embeddingExtent, embeddingMixedExtent, _, vtype)).toList,
        finalNorm = LayerNorm.Params.identity(embeddingExtent, vtype)
      )

class RecordDecoderBlock[Patch: Λ, PatchEmbedding: Λ, Slot: Λ, Embedding: Λ, V: IsFloating](
    patchAxis: Axis[Patch],
    slotAxis: Axis[Slot],
    params: RecordDecoderBlock.Params[PatchEmbedding, Embedding, V]
):

  private val selfAttentionPreNorm = LayerNorm(params.selfAttentionNorm)
  private val crossAttention = MultiHeadFullAttention(patchAxis, slotAxis, params.crossAttention)
  private val crossAttentionPreNorm = LayerNorm(params.crossAttentionNorm)
  private val mlp = MLPEmbeddingMixer(params.mlp)
  private val mlpPreNorm = LayerNorm(params.mlpNorm)

  def apply(
      document: Tensor2[Patch, PatchEmbedding, V],
      sequence: Tensor2[Slot, Embedding, V],
      mask: Tensor2[Slot, Slot, Bool]
  ): Tensor2[Slot, Embedding, V] =
    // The mask is a value rather than a rule, so the attention is built around it here.
    val selfAttention = MultiHeadCustomSelfAttention[Slot, Embedding, V](params.selfAttention, _ => mask, AttentionScore.scaledDotProduct)
    val selfAttended = sequence + selfAttention(sequence.vmap(Axis[Slot])(selfAttentionPreNorm))
    val documentAttended = selfAttended + crossAttention(document, selfAttended.vmap(Axis[Slot])(crossAttentionPreNorm))
    documentAttended + documentAttended.vmap(Axis[Slot])(embedding => mlp(mlpPreNorm(embedding)))

object RecordDecoderBlock:

  case class Params[PatchEmbedding, Embedding, V](
      selfAttention: MultiHeadSelfAttention.Params[Embedding, V],
      selfAttentionNorm: LayerNorm.Params[Embedding, V],
      crossAttention: MultiHeadAttention.Params[PatchEmbedding, Embedding, V],
      crossAttentionNorm: LayerNorm.Params[Embedding, V],
      mlp: MLPEmbeddingMixer.Params[Embedding, V],
      mlpNorm: LayerNorm.Params[Embedding, V]
  )

  object Params:

    def xavierUniformDepthScaled[PatchEmbedding: Λ, Embedding: Λ, V: IsFloating](numBlocks: Int, numHeads: Int, patchEmbeddingExtent: AxisExtent[PatchEmbedding], embeddingExtent: AxisExtent[Embedding], embeddingMixedExtent: AxisExtent[EmbeddingMixed], key: Key, vtype: VType[V] = VType[Float32]): Params[PatchEmbedding, Embedding, V] =
      val (selfKey, crossKey, mlpKey) = key.splitToTuple(3)
      Params(
        selfAttention = MultiHeadSelfAttention.Params.xavierUniformDepthScaled(numBlocks, numHeads, embeddingExtent, selfKey, vtype),
        selfAttentionNorm = LayerNorm.Params.identity(embeddingExtent, vtype),
        crossAttention = MultiHeadAttention.Params.xavierUniformDepthScaled(numBlocks, numHeads, patchEmbeddingExtent, embeddingExtent, crossKey, vtype),
        crossAttentionNorm = LayerNorm.Params.identity(embeddingExtent, vtype),
        mlp = MLPEmbeddingMixer.Params.xavierUniform(embeddingExtent, embeddingMixedExtent, mlpKey, vtype),
        mlpNorm = LayerNorm.Params.identity(embeddingExtent, vtype)
      )
