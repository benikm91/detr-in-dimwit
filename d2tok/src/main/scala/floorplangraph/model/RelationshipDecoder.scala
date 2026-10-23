package d2tok.floorplangraph.model

import d2tok.*
import d2tok.floorplangraph.*
import deepwit.attention.AttentionScore
import deepwit.attention.KVCache
import deepwit.attention.MultiHeadAttention
import deepwit.attention.MultiHeadCausalSelfAttention
import deepwit.attention.MultiHeadCustomAttention
import deepwit.attention.MultiHeadFullAttention
import deepwit.attention.MultiHeadSelfAttention
import deepwit.attention.MultiHeadUnfusedFullAttention
import deepwit.normalization.LayerNorm
import documentEncoder.DocumentEncoder
import documentEncoder.MLPEmbeddingMixer
import dimwit.*
import dimwit.Label as Λ

import scala.language.implicitConversions

/** What a [[RelationshipDecoder]] reads besides the words before it: the document and the nodes,
  * projected once to the keys and values its cross-attentions read, and which of the nodes there are.
  */
case class RelationshipSources[V](document: KVCache[Patch, V], nodes: KVCache[Node, V], present: Tensor1[Node, Bool])

/** A causal transformer over the words of a record's relationships, each reading the document, the
  * nodes and the words before it.
  */
class RelationshipDecoder[V: IsFloating](params: RelationshipDecoder.Params[V]):

  private val blocks = params.blocks.map(RelationshipDecoderBlock(_))
  private val finalNorm = LayerNorm(params.finalNorm)

  def apply(document: Tensor2[Patch, Embedding, V], nodes: Tensor2[Node, Embedding, V], present: Tensor1[Node, Bool], words: Tensor2[RelationshipToken, Embedding, V]): Tensor2[RelationshipToken, Embedding, V] =
    blocks
      .foldLeft(words)((words, block) => block(document, nodes, present, words))
      .vmap(Axis[RelationshipToken])(finalNorm)

  /** Every block's keys and values of the encoded `document` and of the `nodes`, which every
    * decoding step reads.
    */
  def read(document: Tensor2[Patch, Embedding, V], nodes: Tensor2[Node, Embedding, V], present: Tensor1[Node, Bool]): List[RelationshipSources[V]] =
    blocks.map(_.read(document, nodes, present))

  /** Every block's keys and values of `tokens` words, before any is written. */
  def nothingWritten(tokens: AxisExtent[RelationshipToken]): List[KVCache[RelationshipToken, V]] =
    params.blocks.map(block => KVCache.empty(tokens, block.selfAttention))

  /** The `word` embedding at `position` as [[apply]] would make it, and every block's keys and
    * values with it among them for the positions after it to read.
    */
  def decode(sources: List[RelationshipSources[V]], written: List[KVCache[RelationshipToken, V]], position: Tensor0[Int32], word: Tensor1[Embedding, V]): (Tensor1[Embedding, V], List[KVCache[RelationshipToken, V]]) =
    val (decoded, nowWritten) = blocks.lazyZip(sources).lazyZip(written).foldLeft((word, List.empty[KVCache[RelationshipToken, V]])):
      case ((word, nowWritten), (block, sources, written)) =>
        val (decoded, withWord) = block.decode(sources, written, position, word)
        (decoded, nowWritten :+ withWord)
    (finalNorm(decoded), nowWritten)

object RelationshipDecoder:

  case class Params[V](
      blocks: List[RelationshipDecoderBlock.Params[V]],
      finalNorm: LayerNorm.Params[Embedding, V]
  )

  object Params:

    def xavierUniformDepthScaled[V: IsFloating](numBlocks: Int, numHeads: Int, embeddingExtent: AxisExtent[Embedding], embeddingMixedExtent: AxisExtent[DocumentEncoder.EmbeddingMixed], key: Key, vtype: VType[V] = VType[Float32]): Params[V] =
      Params(
        blocks = key.split(numBlocks).map(RelationshipDecoderBlock.Params.xavierUniformDepthScaled(numBlocks, numHeads, embeddingExtent, embeddingMixedExtent, _, vtype)).toList,
        finalNorm = LayerNorm.Params.identity(embeddingExtent, vtype)
      )

/** One block of the [[RelationshipDecoder]]: causal self-attention, then cross-attention onto the
  * encoded document, then cross-attention onto the nodes, then the embedding mixer, each on its own
  * residual branch.
  */
class RelationshipDecoderBlock[V: IsFloating](params: RelationshipDecoderBlock.Params[V]):

  private val selfAttention = MultiHeadCausalSelfAttention(Axis[RelationshipToken], params.selfAttention)
  private val selfAttentionPreNorm = LayerNorm(params.selfAttentionNorm)
  private val documentAttention = MultiHeadFullAttention(Axis[Patch], Axis[RelationshipToken], params.documentAttention)
  private val documentAttentionPreNorm = LayerNorm(params.documentAttentionNorm)
  private val nodeAttentionPreNorm = LayerNorm(params.nodeAttentionNorm)
  private val mlp = MLPEmbeddingMixer(params.mlp)
  private val mlpPreNorm = LayerNorm(params.mlpNorm)

  /** Every word reads the nodes there are, or every node slot where there is none, so that no word
    * reads nothing.
    */
  private def nodeAttention(present: Tensor1[Node, Bool]) =
    val readable = where_!(present.any, present, true)
    MultiHeadCustomAttention[Node, Embedding, RelationshipToken, Embedding, V](
      params.nodeAttention,
      (words, _) => words.vmap(Axis[RelationshipToken])(_ => readable),
      AttentionScore.scaledDotProduct
    )

  def apply(document: Tensor2[Patch, Embedding, V], nodes: Tensor2[Node, Embedding, V], present: Tensor1[Node, Bool], words: Tensor2[RelationshipToken, Embedding, V]): Tensor2[RelationshipToken, Embedding, V] =
    def normed(words: Tensor2[RelationshipToken, Embedding, V], norm: LayerNorm[Embedding, V]) = words.vmap(Axis[RelationshipToken])(norm)
    val afterSelf = words + selfAttention(normed(words, selfAttentionPreNorm))
    val afterDocument = afterSelf + documentAttention(document, normed(afterSelf, documentAttentionPreNorm))
    val afterNodes = afterDocument + nodeAttention(present)(nodes, normed(afterDocument, nodeAttentionPreNorm))
    afterNodes + afterNodes.vmap(Axis[RelationshipToken])(embedding => mlp(mlpPreNorm(embedding)))

  /** The keys and values of the encoded `document` and of the `nodes`, as this block's
    * cross-attentions read them.
    */
  def read(document: Tensor2[Patch, Embedding, V], nodes: Tensor2[Node, Embedding, V], present: Tensor1[Node, Bool]): RelationshipSources[V] =
    RelationshipSources(
      KVCache(document.dot(Axis[Embedding])(params.documentAttention.keyWeights), document.dot(Axis[Embedding])(params.documentAttention.valueWeights)),
      KVCache(nodes.dot(Axis[Embedding])(params.nodeAttention.keyWeights), nodes.dot(Axis[Embedding])(params.nodeAttention.valueWeights)),
      present
    )

  /** The block applied to the `word` embedding at `position`, reading the words `written` before
    * it, the document and the nodes, and `written` with its key and value among them.
    */
  def decode(sources: RelationshipSources[V], written: KVCache[RelationshipToken, V], position: Tensor0[Int32], word: Tensor1[Embedding, V]): (Tensor1[Embedding, V], KVCache[RelationshipToken, V]) =
    val (readWritten, withWord) = selfAttention.decode(written, position, selfAttentionPreNorm(word))
    val afterSelf = word + readWritten
    val at = position.prependAxis(Axis[RelationshipToken])
    def query(embedding: Tensor1[Embedding, V], norm: LayerNorm[Embedding, V], weights: MultiHeadAttention.Params[Embedding, Embedding, V]) =
      norm(embedding).prependAxis(Axis[RelationshipToken]).dot(Axis[Embedding])(weights.queryWeights)
    val documentRead = MultiHeadUnfusedFullAttention[Patch, Embedding, RelationshipToken, Embedding, V](Axis[Patch], Axis[RelationshipToken], params.documentAttention, AttentionScore.scaledDotProduct)
    val patches = Tensor1(Axis[Patch]).fromRange(0 until sources.document.keys.shape(Axis[Patch]))
    val readDocument = documentRead.attendAt(query(afterSelf, documentAttentionPreNorm, params.documentAttention), sources.document.keys, sources.document.values, at, patches)
    val afterDocument = afterSelf + documentRead.projectHeads(readDocument).squeeze(Axis[RelationshipToken])
    val nodeRead = nodeAttention(sources.present)
    val numbers = Tensor1(Axis[Node]).fromRange(0 until sources.nodes.keys.shape(Axis[Node]))
    val readNodes = nodeRead.attendAt(query(afterDocument, nodeAttentionPreNorm, params.nodeAttention), sources.nodes.keys, sources.nodes.values, at, numbers)
    val afterNodes = afterDocument + nodeRead.projectHeads(readNodes).squeeze(Axis[RelationshipToken])
    (afterNodes + mlp(mlpPreNorm(afterNodes)), withWord)

object RelationshipDecoderBlock:

  case class Params[V](
      selfAttention: MultiHeadSelfAttention.Params[Embedding, V],
      selfAttentionNorm: LayerNorm.Params[Embedding, V],
      documentAttention: MultiHeadAttention.Params[Embedding, Embedding, V],
      documentAttentionNorm: LayerNorm.Params[Embedding, V],
      nodeAttention: MultiHeadAttention.Params[Embedding, Embedding, V],
      nodeAttentionNorm: LayerNorm.Params[Embedding, V],
      mlp: MLPEmbeddingMixer.Params[Embedding, V],
      mlpNorm: LayerNorm.Params[Embedding, V]
  )

  object Params:

    def xavierUniformDepthScaled[V: IsFloating](numBlocks: Int, numHeads: Int, embeddingExtent: AxisExtent[Embedding], embeddingMixedExtent: AxisExtent[DocumentEncoder.EmbeddingMixed], key: Key, vtype: VType[V] = VType[Float32]): Params[V] =
      val (selfKey, documentKey, nodeKey, mlpKey) = key.splitToTuple(4)
      Params(
        selfAttention = MultiHeadSelfAttention.Params.xavierUniformDepthScaled(numBlocks, numHeads, embeddingExtent, selfKey, vtype),
        selfAttentionNorm = LayerNorm.Params.identity(embeddingExtent, vtype),
        documentAttention = MultiHeadAttention.Params.xavierUniformDepthScaled(numBlocks, numHeads, embeddingExtent, embeddingExtent, documentKey, vtype),
        documentAttentionNorm = LayerNorm.Params.identity(embeddingExtent, vtype),
        nodeAttention = MultiHeadAttention.Params.xavierUniformDepthScaled(numBlocks, numHeads, embeddingExtent, embeddingExtent, nodeKey, vtype),
        nodeAttentionNorm = LayerNorm.Params.identity(embeddingExtent, vtype),
        mlp = MLPEmbeddingMixer.Params.xavierUniform(embeddingExtent, embeddingMixedExtent, mlpKey, vtype),
        mlpNorm = LayerNorm.Params.identity(embeddingExtent, vtype)
      )
