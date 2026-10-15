package d2seq.model

import d2seq.*
import deepwit.attention.AttentionScore
import deepwit.attention.KVCache
import deepwit.attention.MultiHeadAttention
import deepwit.attention.MultiHeadCausalSelfAttention
import deepwit.attention.MultiHeadFullAttention
import deepwit.attention.MultiHeadSelfAttention
import deepwit.attention.MultiHeadUnfusedFullAttention
import deepwit.normalization.LayerNorm
import deepwit.transformer.CrossTransformerBlock
import documentEncoder.DocumentEncoder
import documentEncoder.MLPEmbeddingMixer
import dimwit.*
import dimwit.Label as Λ

import scala.language.implicitConversions

/** A causal transformer over the nodes, each reading the document and the nodes before it. */
class SequenceDecoder[PatchEmbedding: Λ, Embedding: Λ, V: IsFloating](
    params: SequenceDecoder.Params[PatchEmbedding, Embedding, V]
):

  private val blocks = params.blocks.map(SequenceDecoderBlock(_))
  private val finalNorm = LayerNorm(params.finalNorm)

  def apply(document: Tensor2[Patch, PatchEmbedding, V], nodes: Tensor2[Node, Embedding, V]): Tensor2[Node, Embedding, V] =
    blocks
      .foldLeft(nodes)((nodes, block) => block(document, nodes))
      .vmap(Axis[Node])(finalNorm)

  /** Every block's keys and values of the encoded `document`, which every decoding step reads. */
  def read(document: Tensor2[Patch, PatchEmbedding, V]): List[KVCache[Patch, V]] =
    blocks.map(_.read(document))

  /** Every block's keys and values of `slots` nodes, before any is written. */
  def nothingWritten(slots: AxisExtent[Node]): List[KVCache[Node, V]] =
    params.blocks.map(block => KVCache.empty(slots, block.selfAttention))

  /** The `node` embedding at `slot` as [[apply]] would make it, and every block's keys and values
    * with it among them for the slots after it to read.
    */
  def decode(documentCache: List[KVCache[Patch, V]], written: List[KVCache[Node, V]], slot: Tensor0[Int32], node: Tensor1[Embedding, V]): (Tensor1[Embedding, V], List[KVCache[Node, V]]) =
    val (decoded, nowWritten) = blocks.lazyZip(documentCache).lazyZip(written).foldLeft((node, List.empty[KVCache[Node, V]])):
      case ((node, nowWritten), (block, documentCache, written)) =>
        val (decoded, withNode) = block.decode(documentCache, written, slot, node)
        (decoded, nowWritten :+ withNode)
    (finalNorm(decoded), nowWritten)

object SequenceDecoder:

  case class Params[PatchEmbedding, Embedding, V](
      blocks: List[SequenceDecoderBlock.Params[PatchEmbedding, Embedding, V]],
      finalNorm: LayerNorm.Params[Embedding, V]
  )

  object Params:

    def xavierUniformDepthScaled[PatchEmbedding: Λ, Embedding: Λ, V: IsFloating](numBlocks: Int, numHeads: Int, patchEmbeddingExtent: AxisExtent[PatchEmbedding], embeddingExtent: AxisExtent[Embedding], embeddingMixedExtent: AxisExtent[DocumentEncoder.EmbeddingMixed], key: Key, vtype: VType[V] = VType[Float32]): Params[PatchEmbedding, Embedding, V] =
      Params(
        blocks = key.split(numBlocks).map(SequenceDecoderBlock.Params.xavierUniformDepthScaled(numBlocks, numHeads, patchEmbeddingExtent, embeddingExtent, embeddingMixedExtent, _, vtype)).toList,
        finalNorm = LayerNorm.Params.identity(embeddingExtent, vtype)
      )

/** One block of the [[SequenceDecoder]]: causal self-attention, then cross-attention onto the
  * encoded document, then the embedding mixer, each on its own residual branch.
  */
class SequenceDecoderBlock[PatchEmbedding: Λ, Embedding: Λ, V: IsFloating](
    params: SequenceDecoderBlock.Params[PatchEmbedding, Embedding, V]
) extends CrossTransformerBlock[Patch, PatchEmbedding, Node, Embedding, V](Axis[Patch], Axis[Node]):

  private val selfAttention = MultiHeadCausalSelfAttention(Axis[Node], params.selfAttention)
  private val selfAttentionPreNorm = LayerNorm(params.selfAttentionNorm)
  private val documentAttention = MultiHeadFullAttention(Axis[Patch], Axis[Node], params.documentAttention)
  private val documentAttentionPreNorm = LayerNorm(params.documentAttentionNorm)
  private val mlp = MLPEmbeddingMixer(params.mlp)
  private val mlpPreNorm = LayerNorm(params.mlpNorm)

  override protected def contextMixer(nodes: Tensor2[Node, Embedding, V]): Tensor2[Node, Embedding, V] =
    selfAttention(nodes.vmap(Axis[Node])(selfAttentionPreNorm))

  override protected def crossContextMixer(document: Tensor2[Patch, PatchEmbedding, V], nodes: Tensor2[Node, Embedding, V]): Tensor2[Node, Embedding, V] =
    documentAttention(document, nodes.vmap(Axis[Node])(documentAttentionPreNorm))

  override protected def embeddingMixer(embedding: Tensor1[Embedding, V]): Tensor1[Embedding, V] =
    mlp(mlpPreNorm(embedding))

  /** The keys and values of the encoded `document`, as this block's cross-attention reads them. */
  def read(document: Tensor2[Patch, PatchEmbedding, V]): KVCache[Patch, V] =
    KVCache(
      document.dot(Axis[PatchEmbedding])(params.documentAttention.keyWeights),
      document.dot(Axis[PatchEmbedding])(params.documentAttention.valueWeights)
    )

  /** The block applied to the `node` embedding at `slot`, reading the nodes `written` before it and
    * the document, and `written` with its key and value among them.
    */
  def decode(documentCache: KVCache[Patch, V], written: KVCache[Node, V], slot: Tensor0[Int32], node: Tensor1[Embedding, V]): (Tensor1[Embedding, V], KVCache[Node, V]) =
    val (readWritten, withNode) = selfAttention.decode(written, slot, selfAttentionPreNorm(node))
    val afterSelf = node + readWritten
    val documentRead = MultiHeadUnfusedFullAttention[Patch, PatchEmbedding, Node, Embedding, V](Axis[Patch], Axis[Node], params.documentAttention, AttentionScore.scaledDotProduct)
    val documentQuery = documentAttentionPreNorm(afterSelf).prependAxis(Axis[Node]).dot(Axis[Embedding])(params.documentAttention.queryWeights)
    val patches = Tensor1(Axis[Patch]).fromRange(0 until documentCache.keys.shape(Axis[Patch]))
    val readDocument = documentRead.projectHeads(documentRead.attendAt(documentQuery, documentCache.keys, documentCache.values, slot.prependAxis(Axis[Node]), patches))
    val afterDocument = afterSelf + readDocument.squeeze(Axis[Node])
    (afterDocument + mlp(mlpPreNorm(afterDocument)), withNode)

object SequenceDecoderBlock:

  case class Params[PatchEmbedding, Embedding, V](
      selfAttention: MultiHeadSelfAttention.Params[Embedding, V],
      selfAttentionNorm: LayerNorm.Params[Embedding, V],
      documentAttention: MultiHeadAttention.Params[PatchEmbedding, Embedding, V],
      documentAttentionNorm: LayerNorm.Params[Embedding, V],
      mlp: MLPEmbeddingMixer.Params[Embedding, V],
      mlpNorm: LayerNorm.Params[Embedding, V]
  )

  object Params:

    def xavierUniformDepthScaled[PatchEmbedding: Λ, Embedding: Λ, V: IsFloating](numBlocks: Int, numHeads: Int, patchEmbeddingExtent: AxisExtent[PatchEmbedding], embeddingExtent: AxisExtent[Embedding], embeddingMixedExtent: AxisExtent[DocumentEncoder.EmbeddingMixed], key: Key, vtype: VType[V] = VType[Float32]): Params[PatchEmbedding, Embedding, V] =
      val (selfKey, documentKey, mlpKey) = key.splitToTuple(3)
      Params(
        selfAttention = MultiHeadSelfAttention.Params.xavierUniformDepthScaled(numBlocks, numHeads, embeddingExtent, selfKey, vtype),
        selfAttentionNorm = LayerNorm.Params.identity(embeddingExtent, vtype),
        documentAttention = MultiHeadAttention.Params.xavierUniformDepthScaled(numBlocks, numHeads, patchEmbeddingExtent, embeddingExtent, documentKey, vtype),
        documentAttentionNorm = LayerNorm.Params.identity(embeddingExtent, vtype),
        mlp = MLPEmbeddingMixer.Params.xavierUniform(embeddingExtent, embeddingMixedExtent, mlpKey, vtype),
        mlpNorm = LayerNorm.Params.identity(embeddingExtent, vtype)
      )
