package d2g.model

import d2g.*
import d2g.train.*
import d2g.eval.*
import d2g.config.*
import d2s.model.*
import deepwit.activation.gelu
import deepwit.attention.AttentionMask
import deepwit.attention.AttentionScore
import deepwit.attention.KVCache
import deepwit.attention.MultiHeadAttention
import deepwit.attention.MultiHeadCustomAttention
import deepwit.attention.MultiHeadCustomSelfAttention
import deepwit.attention.MultiHeadFullAttention
import deepwit.attention.MultiHeadFullSelfAttention
import deepwit.attention.MultiHeadSelfAttention
import deepwit.attention.MultiHeadUnfusedFullAttention
import deepwit.base.AffineLayer
import deepwit.normalization.LayerNorm
import deepwit.transformer.TransformerBlock
import dimwit.*
import dimwit.Label as Λ

import scala.language.implicitConversions

/** Every [[PoolQuery]]'s answer for every [[Edge]] slot. */
type EdgePrediction = PoolQuery |*| Edge

/** What an [[EdgeDecoder]] attends onto: the nodes as a [[NodeDecoder]] carries them, and which of
  * the slots hold a node at all. The positions past a record are there to make every record the
  * same shape and relate nothing, so the two always travel together.
  */
case class NodeSource[Embedding, V](nodeEmbeddings: Tensor2[Node, Embedding, V], presentMask: Tensor1[Node, Bool])

/** What a block of the [[EdgeDecoder]] reads besides the relationships taken: the document and the
  * nodes, projected once to the keys and values its cross-attentions read, and which of the nodes
  * there are.
  */
case class EdgeSources[V](documentCache: KVCache[Patch, V], nodeCache: KVCache[Node, V], presentMask: Tensor1[Node, Bool])

/** The decoder of the record's relationships, read the same way as the [[NodeDecoder]].
  *
  * A relationship is a pair of nodes, so this decoder reads the nodes the [[NodeDecoder]] made of
  * the record as well as the document — the nodes it is given are the taken ones, which are a
  * record, and not the predictions, which are guesses and may hold a node twice or not at all.
  */
class EdgeDecoder[PatchEmbedding: Λ, Embedding: Λ, V: IsFloating](
    params: EdgeDecoder.Params[PatchEmbedding, Embedding, V]
):

  private val blocks = params.blocks.map(EdgeDecoderBlock(_))
  private val finalNorm = LayerNorm(params.finalNorm)

  /** The taken relationships as the decoder carries them, and what each asked query would answer at
    * every slot.
    */
  def forTraining(
      document: Tensor2[Patch, PatchEmbedding, V],
      nodes: NodeSource[Embedding, V],
      edges: Tensor2[Edge, Embedding, V],
      asked: Tensor3[PoolQuery, Edge, Embedding, V]
  ): (Tensor2[Edge, Embedding, V], Tensor3[PoolQuery, Edge, Embedding, V]) =
    val perQuery = Shape2(asked.shape.extent(Axis[PoolQuery]), asked.shape.extent(Axis[Edge]))
    val predictions = asked.flatten((Axis[PoolQuery], Axis[Edge]))
    val (carried, answered) = blocks.foldLeft((edges, predictions)):
      case ((edges, predictions), block) => block.forTraining(document, nodes, edges, predictions)
    (
      carried.vmap(Axis[Edge])(finalNorm),
      answered.vmap(Axis[EdgePrediction])(finalNorm).unflatten(Axis[EdgePrediction], perQuery)
    )

  /** Every block's keys and values of the encoded `document` and of the `nodes`, which every
    * decoding step reads.
    */
  def read(document: Tensor2[Patch, PatchEmbedding, V], nodes: NodeSource[Embedding, V]): List[EdgeSources[V]] =
    blocks.map(_.read(document, nodes))

  /** Every block's keys and values of `slots` taken relationships, before any is taken. */
  def nothingTaken(slots: AxisExtent[Edge]): List[KVCache[Edge, V]] =
    params.blocks.map(block => KVCache.empty(slots, block.selfAttention))

  /** Takes the `edge` embedding at `slot`: every block's keys and values of the taken
    * relationships, with it among them for the slots after it to read.
    */
  def take(sources: List[EdgeSources[V]], taken: List[KVCache[Edge, V]], slot: Tensor0[Int32], edge: Tensor1[Embedding, V]): List[KVCache[Edge, V]] =
    val (_, nowTaken) = blocks.lazyZip(sources).lazyZip(taken).foldLeft((edge, List.empty[KVCache[Edge, V]])):
      case ((edge, nowTaken), (block, sources, taken)) =>
        val (carried, withEdge) = block.take(sources, taken, slot, edge)
        (carried, nowTaken :+ withEdge)
    nowTaken

  /** What each of the `asked` queries answers at `slot`, given the relationships taken before it,
    * as [[forTraining]] would answer there.
    */
  def answer(sources: List[EdgeSources[V]], taken: List[KVCache[Edge, V]], slot: Tensor0[Int32], asked: Tensor2[PoolQuery, Embedding, V]): Tensor2[PoolQuery, Embedding, V] =
    val answered = blocks.lazyZip(sources).lazyZip(taken).foldLeft(asked):
      case (asked, (block, sources, taken)) => block.answer(sources, taken, slot, asked)
    answered.vmap(Axis[PoolQuery])(finalNorm)

object EdgeDecoder:

  /** The sequence a block decodes: every relationship embedding, then every prediction embedding. */
  type Context = Edge |+| EdgePrediction

  case class Params[PatchEmbedding, Embedding, V](
      blocks: List[EdgeDecoderBlock.Params[PatchEmbedding, Embedding, V]],
      finalNorm: LayerNorm.Params[Embedding, V]
  )

  object Params:

    def xavierUniformDepthScaled[PatchEmbedding: Λ, Embedding: Λ, V: IsFloating](numBlocks: Int, numHeads: Int, patchEmbeddingExtent: AxisExtent[PatchEmbedding], embeddingExtent: AxisExtent[Embedding], embeddingMixedExtent: AxisExtent[EmbeddingMixed], key: Key, vtype: VType[V] = VType[Float32]): Params[PatchEmbedding, Embedding, V] =
      Params(
        blocks = key.split(numBlocks).map(EdgeDecoderBlock.Params.xavierUniformDepthScaled(numBlocks, numHeads, patchEmbeddingExtent, embeddingExtent, embeddingMixedExtent, _, vtype)).toList,
        finalNorm = LayerNorm.Params.identity(embeddingExtent, vtype)
      )

/** One block of the [[EdgeDecoder]]: masked self-attention, then cross-attention onto the encoded
  * document, then cross-attention onto the taken nodes, then the embedding mixer, each on its own
  * residual branch.
  */
class EdgeDecoderBlock[PatchEmbedding: Λ, Embedding: Λ, V: IsFloating](
    params: EdgeDecoderBlock.Params[PatchEmbedding, Embedding, V]
):

  import EdgeDecoder.Context

  private val contextAxis = Axis[Context]

  private def jointAttention(slots: Int) = MultiHeadCustomSelfAttention(
    params.selfAttention,
    jointSequenceMask[Context, Context](slots),
    AttentionScore.scaledDotProduct
  )
  private val selfAttentionPreNorm = LayerNorm(params.selfAttentionNorm)
  private val documentAttention = MultiHeadFullAttention(Axis[Patch], contextAxis, params.documentAttention)
  private val documentAttentionPreNorm = LayerNorm(params.documentAttentionNorm)
  private val nodeAttentionPreNorm = LayerNorm(params.nodeAttentionNorm)
  private val mlp = MLPEmbeddingMixer(params.mlp)
  private val mlpPreNorm = LayerNorm(params.mlpNorm)

  def forTraining(
      document: Tensor2[Patch, PatchEmbedding, V],
      nodes: NodeSource[Embedding, V],
      edges: Tensor2[Edge, Embedding, V],
      predictions: Tensor2[EdgePrediction, Embedding, V]
  ): (Tensor2[Edge, Embedding, V], Tensor2[EdgePrediction, Embedding, V]) =
    var x = concatenate(edges, predictions)
    x = x + jointAttention(edges.shape(Axis[Edge]))(x.vmap(contextAxis)(selfAttentionPreNorm))
    x = x + documentAttention(document, x.vmap(contextAxis)(documentAttentionPreNorm))
    x = x + nodeAttention[Context](nodes.presentMask)(nodes.nodeEmbeddings, x.vmap(contextAxis)(nodeAttentionPreNorm))
    x = x + x.vmap(contextAxis)(embedding => mlp(mlpPreNorm(embedding)))
    x.deconcatenate(contextAxis, (edges.extent(Axis[Edge]), predictions.extent(Axis[EdgePrediction])))

  /** Every row reads the nodes there are, or every slot where there is none, so that no row reads
    * nothing.
    */
  private def nodeAttention[Row: Λ](presentMask: Tensor1[Node, Bool]) =
    val readable = where_!(presentMask.any, presentMask, true)
    val mask: AttentionMask[Row, Node] = (rows, _) => rows.vmap(Axis[Row])(_ => readable)
    MultiHeadCustomAttention[Node, Embedding, Row, Embedding, V](params.nodeAttention, mask, AttentionScore.scaledDotProduct)

  // Decoding one slot at a time, as the NodeDecoderBlock does, with the taken nodes read from a
  // cache of their own.

  private val selfWeights = params.selfAttention.multiHeadAttention

  /** The keys and values of the encoded `document` and of the `nodes`, as this block's
    * cross-attentions read them.
    */
  def read(document: Tensor2[Patch, PatchEmbedding, V], nodes: NodeSource[Embedding, V]): EdgeSources[V] =
    EdgeSources(
      KVCache(
        document.dot(Axis[PatchEmbedding])(params.documentAttention.keyWeights),
        document.dot(Axis[PatchEmbedding])(params.documentAttention.valueWeights)
      ),
      KVCache(
        nodes.nodeEmbeddings.dot(Axis[Embedding])(params.nodeAttention.keyWeights),
        nodes.nodeEmbeddings.dot(Axis[Embedding])(params.nodeAttention.valueWeights)
      ),
      nodes.presentMask
    )

  /** Takes the `edge` embedding at `slot`: what the block makes of it, and `taken` with its key and
    * value written in.
    */
  def take(sources: EdgeSources[V], taken: KVCache[Edge, V], slot: Tensor0[Int32], edge: Tensor1[Embedding, V]): (Tensor1[Embedding, V], KVCache[Edge, V]) =
    val normed = selfAttentionPreNorm(edge)
    val withEdge = KVCache(
      taken.keys.set(Axis[Edge].at(slot))(normed.dot(Axis[Embedding])(selfWeights.keyWeights)),
      taken.values.set(Axis[Edge].at(slot))(normed.dot(Axis[Embedding])(selfWeights.valueWeights))
    )
    val slots = taken.keys.shape.extent(Axis[Edge])
    val carried = decoded(
      sources,
      edge.prependAxis(Axis[Taking]),
      normed.prependAxis(Axis[Taking]),
      withEdge,
      slot.prependAxis(Axis[Taking]),
      Tensor1(Axis[Edge]).fromRange(0 until slots.size),
      slots.size
    )
    (carried.squeeze(Axis[Taking]), withEdge)

  /** What the `asked` queries make of their prediction embeddings at `slot`, reading the
    * relationships taken before it and themselves.
    */
  def answer(sources: EdgeSources[V], taken: KVCache[Edge, V], slot: Tensor0[Int32], asked: Tensor2[PoolQuery, Embedding, V]): Tensor2[PoolQuery, Embedding, V] =
    val slots = taken.keys.shape(Axis[Edge])
    val normed = asked.vmap(Axis[PoolQuery])(selfAttentionPreNorm)
    val queries = Tensor1(Axis[PoolQuery]).fromRange(0 until asked.shape(Axis[PoolQuery]))
    val positions = queries *! slots +! slots +! slot
    decoded(
      sources,
      asked,
      normed,
      KVCache(
        concatenate(taken.keys, normed.dot(Axis[Embedding])(selfWeights.keyWeights)),
        concatenate(taken.values, normed.dot(Axis[Embedding])(selfWeights.valueWeights))
      ),
      positions,
      concatenate(Tensor1(Axis[Edge]).fromRange(0 until slots), positions),
      slots
    )

  /** The block applied to `rows` at `positions` of the joined sequence, attending to the `taken`
    * relationships at `takenPositions`, to the document and to the nodes.
    */
  private def decoded[Row: Λ, Source: Λ](
      sources: EdgeSources[V],
      rows: Tensor2[Row, Embedding, V],
      normed: Tensor2[Row, Embedding, V],
      taken: KVCache[Source, V],
      positions: Tensor1[Row, Int32],
      takenPositions: Tensor1[Source, Int32],
      slots: Int
  ): Tensor2[Row, Embedding, V] =
    val selfAttention = MultiHeadCustomAttention[Source, Embedding, Row, Embedding, V](selfWeights, jointSequenceMask[Row, Source](slots), AttentionScore.scaledDotProduct)
    val afterSelf = rows + selfAttention.projectHeads(selfAttention.attendAt(normed.dot(Axis[Embedding])(selfWeights.queryWeights), taken.keys, taken.values, positions, takenPositions))
    val documentRead = MultiHeadUnfusedFullAttention[Patch, PatchEmbedding, Row, Embedding, V](Axis[Patch], Axis[Row], params.documentAttention, AttentionScore.scaledDotProduct)
    val documentQueries = afterSelf.vmap(Axis[Row])(documentAttentionPreNorm).dot(Axis[Embedding])(params.documentAttention.queryWeights)
    val patches = Tensor1(Axis[Patch]).fromRange(0 until sources.documentCache.keys.shape(Axis[Patch]))
    val afterDocument = afterSelf + documentRead.projectHeads(documentRead.attendAt(documentQueries, sources.documentCache.keys, sources.documentCache.values, positions, patches))
    val nodeRead = nodeAttention[Row](sources.presentMask)
    val nodeQueries = afterDocument.vmap(Axis[Row])(nodeAttentionPreNorm).dot(Axis[Embedding])(params.nodeAttention.queryWeights)
    val nodes = Tensor1(Axis[Node]).fromRange(0 until sources.nodeCache.keys.shape(Axis[Node]))
    val afterNodes = afterDocument + nodeRead.projectHeads(nodeRead.attendAt(nodeQueries, sources.nodeCache.keys, sources.nodeCache.values, positions, nodes))
    afterNodes + afterNodes.vmap(Axis[Row])(embedding => mlp(mlpPreNorm(embedding)))

/** Axis of the one relationship a decoding step takes. */
private trait Taking derives Label

object EdgeDecoderBlock:

  case class Params[PatchEmbedding, Embedding, V](
      selfAttention: MultiHeadSelfAttention.Params[Embedding, V],
      selfAttentionNorm: LayerNorm.Params[Embedding, V],
      documentAttention: MultiHeadAttention.Params[PatchEmbedding, Embedding, V],
      documentAttentionNorm: LayerNorm.Params[Embedding, V],
      nodeAttention: MultiHeadAttention.Params[Embedding, Embedding, V],
      nodeAttentionNorm: LayerNorm.Params[Embedding, V],
      mlp: MLPEmbeddingMixer.Params[Embedding, V],
      mlpNorm: LayerNorm.Params[Embedding, V]
  )

  object Params:

    def xavierUniformDepthScaled[PatchEmbedding: Λ, Embedding: Λ, V: IsFloating](numBlocks: Int, numHeads: Int, patchEmbeddingExtent: AxisExtent[PatchEmbedding], embeddingExtent: AxisExtent[Embedding], embeddingMixedExtent: AxisExtent[EmbeddingMixed], key: Key, vtype: VType[V] = VType[Float32]): Params[PatchEmbedding, Embedding, V] =
      val (selfKey, documentKey, nodeKey, mlpKey) = key.splitToTuple(4)
      Params(
        selfAttention = MultiHeadSelfAttention.Params.xavierUniformDepthScaled(numBlocks, numHeads, embeddingExtent, selfKey, vtype),
        selfAttentionNorm = LayerNorm.Params.identity(embeddingExtent, vtype),
        documentAttention = MultiHeadAttention.Params.xavierUniformDepthScaled(numBlocks, numHeads, patchEmbeddingExtent, embeddingExtent, documentKey, vtype),
        documentAttentionNorm = LayerNorm.Params.identity(embeddingExtent, vtype),
        nodeAttention = MultiHeadAttention.Params.xavierUniformDepthScaled(numBlocks, numHeads, embeddingExtent, embeddingExtent, nodeKey, vtype),
        nodeAttentionNorm = LayerNorm.Params.identity(embeddingExtent, vtype),
        mlp = MLPEmbeddingMixer.Params.init(embeddingExtent, embeddingMixedExtent, mlpKey, vtype),
        mlpNorm = LayerNorm.Params.identity(embeddingExtent, vtype)
      )
