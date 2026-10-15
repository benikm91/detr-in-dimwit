package d2tok.model

import d2tok.*
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

/** A causal transformer over the words, each reading the document and the words before it. */
class WordDecoder[PatchEmbedding: Λ, Embedding: Λ, V: IsFloating](
    params: WordDecoder.Params[PatchEmbedding, Embedding, V]
):

  private val blocks = params.blocks.map(WordDecoderBlock(_))
  private val finalNorm = LayerNorm(params.finalNorm)

  def apply(document: Tensor2[Patch, PatchEmbedding, V], words: Tensor2[Token, Embedding, V]): Tensor2[Token, Embedding, V] =
    blocks
      .foldLeft(words)((words, block) => block(document, words))
      .vmap(Axis[Token])(finalNorm)

  /** Every block's keys and values of the encoded `document`, which every decoding step reads. */
  def read(document: Tensor2[Patch, PatchEmbedding, V]): List[KVCache[Patch, V]] =
    blocks.map(_.read(document))

  /** Every block's keys and values of `tokens` words, before any is written. */
  def nothingWritten(tokens: AxisExtent[Token]): List[KVCache[Token, V]] =
    params.blocks.map(block => KVCache.empty(tokens, block.selfAttention))

  /** The `word` embedding at `position` as [[apply]] would make it, and every block's keys and
    * values with it among them for the positions after it to read.
    */
  def decode(documentCache: List[KVCache[Patch, V]], written: List[KVCache[Token, V]], position: Tensor0[Int32], word: Tensor1[Embedding, V]): (Tensor1[Embedding, V], List[KVCache[Token, V]]) =
    val (decoded, nowWritten) = blocks.lazyZip(documentCache).lazyZip(written).foldLeft((word, List.empty[KVCache[Token, V]])):
      case ((word, nowWritten), (block, documentCache, written)) =>
        val (decoded, withWord) = block.decode(documentCache, written, position, word)
        (decoded, nowWritten :+ withWord)
    (finalNorm(decoded), nowWritten)

object WordDecoder:

  case class Params[PatchEmbedding, Embedding, V](
      blocks: List[WordDecoderBlock.Params[PatchEmbedding, Embedding, V]],
      finalNorm: LayerNorm.Params[Embedding, V]
  )

  object Params:

    def xavierUniformDepthScaled[PatchEmbedding: Λ, Embedding: Λ, V: IsFloating](numBlocks: Int, numHeads: Int, patchEmbeddingExtent: AxisExtent[PatchEmbedding], embeddingExtent: AxisExtent[Embedding], embeddingMixedExtent: AxisExtent[DocumentEncoder.EmbeddingMixed], key: Key, vtype: VType[V] = VType[Float32]): Params[PatchEmbedding, Embedding, V] =
      Params(
        blocks = key.split(numBlocks).map(WordDecoderBlock.Params.xavierUniformDepthScaled(numBlocks, numHeads, patchEmbeddingExtent, embeddingExtent, embeddingMixedExtent, _, vtype)).toList,
        finalNorm = LayerNorm.Params.identity(embeddingExtent, vtype)
      )

/** One block of the [[WordDecoder]]: causal self-attention, then cross-attention onto the encoded
  * document, then the embedding mixer, each on its own residual branch.
  */
class WordDecoderBlock[PatchEmbedding: Λ, Embedding: Λ, V: IsFloating](
    params: WordDecoderBlock.Params[PatchEmbedding, Embedding, V]
) extends CrossTransformerBlock[Patch, PatchEmbedding, Token, Embedding, V](Axis[Patch], Axis[Token]):

  private val selfAttention = MultiHeadCausalSelfAttention(Axis[Token], params.selfAttention)
  private val selfAttentionPreNorm = LayerNorm(params.selfAttentionNorm)
  private val documentAttention = MultiHeadFullAttention(Axis[Patch], Axis[Token], params.documentAttention)
  private val documentAttentionPreNorm = LayerNorm(params.documentAttentionNorm)
  private val mlp = MLPEmbeddingMixer(params.mlp)
  private val mlpPreNorm = LayerNorm(params.mlpNorm)

  override protected def contextMixer(words: Tensor2[Token, Embedding, V]): Tensor2[Token, Embedding, V] =
    selfAttention(words.vmap(Axis[Token])(selfAttentionPreNorm))

  override protected def crossContextMixer(document: Tensor2[Patch, PatchEmbedding, V], words: Tensor2[Token, Embedding, V]): Tensor2[Token, Embedding, V] =
    documentAttention(document, words.vmap(Axis[Token])(documentAttentionPreNorm))

  override protected def embeddingMixer(embedding: Tensor1[Embedding, V]): Tensor1[Embedding, V] =
    mlp(mlpPreNorm(embedding))

  /** The keys and values of the encoded `document`, as this block's cross-attention reads them. */
  def read(document: Tensor2[Patch, PatchEmbedding, V]): KVCache[Patch, V] =
    KVCache(
      document.dot(Axis[PatchEmbedding])(params.documentAttention.keyWeights),
      document.dot(Axis[PatchEmbedding])(params.documentAttention.valueWeights)
    )

  /** The block applied to the `word` embedding at `position`, reading the words `written` before
    * it and the document, and `written` with its key and value among them.
    */
  def decode(documentCache: KVCache[Patch, V], written: KVCache[Token, V], position: Tensor0[Int32], word: Tensor1[Embedding, V]): (Tensor1[Embedding, V], KVCache[Token, V]) =
    val (readWritten, withWord) = selfAttention.decode(written, position, selfAttentionPreNorm(word))
    val afterSelf = word + readWritten
    val documentRead = MultiHeadUnfusedFullAttention[Patch, PatchEmbedding, Token, Embedding, V](Axis[Patch], Axis[Token], params.documentAttention, AttentionScore.scaledDotProduct)
    val documentQuery = documentAttentionPreNorm(afterSelf).prependAxis(Axis[Token]).dot(Axis[Embedding])(params.documentAttention.queryWeights)
    val patches = Tensor1(Axis[Patch]).fromRange(0 until documentCache.keys.shape(Axis[Patch]))
    val readDocument = documentRead.projectHeads(documentRead.attendAt(documentQuery, documentCache.keys, documentCache.values, position.prependAxis(Axis[Token]), patches))
    val afterDocument = afterSelf + readDocument.squeeze(Axis[Token])
    (afterDocument + mlp(mlpPreNorm(afterDocument)), withWord)

object WordDecoderBlock:

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
