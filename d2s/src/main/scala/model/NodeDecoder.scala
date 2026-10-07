package d2s.model

import d2s.*
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

/** Queries from the query pool to learn different remaining nodes/edges. */
trait PoolQuery derives Label

/** Every [[PoolQuery]]'s answer for every [[Node]] slot. */
type NodePrediction = PoolQuery |*| Node

/** The decoder of the record's nodes.
  *
  * An embedding in a decoder has two jobs — become what its slot predicts, and keep carrying what
  * its slot holds for the others to read. Remaining-node prediction cannot do both at once, since
  * a later slot has to know what is taken already in order to answer with something else. So every
  * slot gets both: a *node embedding* carrying the taken node, and a *prediction embedding*
  * becoming one of the remaining ones.
  *
  * It takes no mask. What a slot may read is not the caller's to say.
  */
class NodeDecoder[PatchEmbedding: Λ, Embedding: Λ, V: IsFloating](
    params: NodeDecoder.Params[PatchEmbedding, Embedding, V]
):

  private val blocks = params.blocks.map(NodeDecoderBlock(_))
  private val finalNorm = LayerNorm(params.finalNorm)

  /** The taken nodes as the decoder carries them, and what each asked query would answer at every
    * slot.
    */
  def forTraining(
      document: Tensor2[Patch, PatchEmbedding, V],
      nodes: Tensor2[Node, Embedding, V],
      asked: Tensor3[PoolQuery, Node, Embedding, V]
  ): (Tensor2[Node, Embedding, V], Tensor3[PoolQuery, Node, Embedding, V]) =
    val perQuery = Shape2(asked.shape.extent(Axis[PoolQuery]), asked.shape.extent(Axis[Node]))
    val predictions = asked.flatten((Axis[PoolQuery], Axis[Node]))
    val (carried, answered) = blocks.foldLeft((nodes, predictions)):
      case ((nodes, predictions), block) => block.forTraining(document, nodes, predictions)
    (
      carried.vmap(Axis[Node])(finalNorm),
      answered.vmap(Axis[NodePrediction])(finalNorm).unflatten(Axis[NodePrediction], perQuery)
    )

  /** Every block's keys and values of the encoded `document`, which every decoding step reads. */
  def read(document: Tensor2[Patch, PatchEmbedding, V]): List[KVCache[Patch, V]] =
    blocks.map(_.read(document))

  /** Every block's keys and values of `slots` taken nodes, before any is taken. */
  def nothingTaken(slots: AxisExtent[Node]): List[KVCache[Node, V]] =
    params.blocks.map(block => KVCache.empty(slots, block.selfAttention))

  /** Takes the `node` embedding at `slot`: every block's keys and values of the taken nodes, with
    * it among them for the slots after it to read.
    */
  def take(document: List[KVCache[Patch, V]], taken: List[KVCache[Node, V]], slot: Tensor0[Int32], node: Tensor1[Embedding, V]): List[KVCache[Node, V]] =
    val (_, nowTaken) = blocks.lazyZip(document).lazyZip(taken).foldLeft((node, List.empty[KVCache[Node, V]])):
      case ((node, nowTaken), (block, document, taken)) =>
        val (carried, withNode) = block.take(document, taken, slot, node)
        (carried, nowTaken :+ withNode)
    nowTaken

  /** What each of the `asked` queries answers at `slot`, given the nodes taken before it, as
    * [[forTraining]] would answer there.
    */
  def answer(document: List[KVCache[Patch, V]], taken: List[KVCache[Node, V]], slot: Tensor0[Int32], asked: Tensor2[PoolQuery, Embedding, V]): Tensor2[PoolQuery, Embedding, V] =
    val answered = blocks.lazyZip(document).lazyZip(taken).foldLeft(asked):
      case (asked, (block, document, taken)) => block.answer(document, taken, slot, asked)
    answered.vmap(Axis[PoolQuery])(finalNorm)

object NodeDecoder:

  /** The sequence a block decodes: every node embedding, then every prediction embedding. */
  type Context = Node |+| NodePrediction

  case class Params[PatchEmbedding, Embedding, V](
      blocks: List[NodeDecoderBlock.Params[PatchEmbedding, Embedding, V]],
      finalNorm: LayerNorm.Params[Embedding, V]
  )

  object Params:

    def xavierUniformDepthScaled[PatchEmbedding: Λ, Embedding: Λ, V: IsFloating](numBlocks: Int, numHeads: Int, patchEmbeddingExtent: AxisExtent[PatchEmbedding], embeddingExtent: AxisExtent[Embedding], embeddingMixedExtent: AxisExtent[EmbeddingMixed], key: Key, vtype: VType[V] = VType[Float32]): Params[PatchEmbedding, Embedding, V] =
      Params(
        blocks = key.split(numBlocks).map(NodeDecoderBlock.Params.xavierUniformDepthScaled(numBlocks, numHeads, patchEmbeddingExtent, embeddingExtent, embeddingMixedExtent, _, vtype)).toList,
        finalNorm = LayerNorm.Params.identity(embeddingExtent, vtype)
      )

/** One block of the [[NodeDecoder]]: masked self-attention, then cross-attention onto the encoded
  * document, then the embedding mixer, each on its own residual branch.
  *
  * The halves it is given are joined into one sequence and taken apart again, so how they are laid
  * out, and the mask that goes with the layout, stay in this class.
  */
class NodeDecoderBlock[PatchEmbedding: Λ, Embedding: Λ, V: IsFloating](
    params: NodeDecoderBlock.Params[PatchEmbedding, Embedding, V]
):

  import NodeDecoder.Context

  private val contextAxis = Axis[Context]

  private def jointAttention(slots: Int) = MultiHeadCustomSelfAttention(
    params.selfAttention,
    jointSequenceMask[Context, Context](slots),
    AttentionScore.scaledDotProduct
  )
  private val selfAttentionPreNorm = LayerNorm(params.selfAttentionNorm)
  private val documentAttention = MultiHeadFullAttention(Axis[Patch], contextAxis, params.documentAttention)
  private val documentAttentionPreNorm = LayerNorm(params.documentAttentionNorm)
  private val mlp = MLPEmbeddingMixer(params.mlp)
  private val mlpPreNorm = LayerNorm(params.mlpNorm)

  def forTraining(
      document: Tensor2[Patch, PatchEmbedding, V],
      nodes: Tensor2[Node, Embedding, V],
      predictions: Tensor2[NodePrediction, Embedding, V]
  ): (Tensor2[Node, Embedding, V], Tensor2[NodePrediction, Embedding, V]) =
    var x = concatenate(nodes, predictions)
    x = x + jointAttention(nodes.shape(Axis[Node]))(x.vmap(contextAxis)(selfAttentionPreNorm))
    x = x + documentAttention(document, x.vmap(contextAxis)(documentAttentionPreNorm))
    x = x + x.vmap(contextAxis)(embedding => mlp(mlpPreNorm(embedding)))
    x.deconcatenate(contextAxis, (nodes.extent(Axis[Node]), predictions.extent(Axis[NodePrediction])))

  // Decoding one slot at a time: the keys and values of the document and of the taken nodes are
  // kept, and a step projects only the embeddings it adds. Each slot sits where it sits in the
  // joined sequence, so the same mask decides what it reads.

  private val selfWeights = params.selfAttention.multiHeadAttention

  /** The keys and values of the encoded `document`, as this block's cross-attention reads them. */
  def read(document: Tensor2[Patch, PatchEmbedding, V]): KVCache[Patch, V] =
    KVCache(
      document.dot(Axis[PatchEmbedding])(params.documentAttention.keyWeights),
      document.dot(Axis[PatchEmbedding])(params.documentAttention.valueWeights)
    )

  /** Takes the `node` embedding at `slot`: what the block makes of it, and `taken` with its key and
    * value written in.
    */
  def take(document: KVCache[Patch, V], taken: KVCache[Node, V], slot: Tensor0[Int32], node: Tensor1[Embedding, V]): (Tensor1[Embedding, V], KVCache[Node, V]) =
    val normed = selfAttentionPreNorm(node)
    val withNode = KVCache(
      taken.keys.set(Axis[Node].at(slot))(normed.dot(Axis[Embedding])(selfWeights.keyWeights)),
      taken.values.set(Axis[Node].at(slot))(normed.dot(Axis[Embedding])(selfWeights.valueWeights))
    )
    val slots = taken.keys.shape.extent(Axis[Node])
    val carried = decoded(
      document,
      stack(Seq(node), Axis[Taking]),
      stack(Seq(normed), Axis[Taking]),
      withNode,
      stack(Seq(slot), Axis[Taking]),
      Tensor1(Axis[Node]).fromRange(0 until slots.size),
      slots.size
    )
    (carried.slice(Axis[Taking].at(0)), withNode)

  /** What the `asked` queries make of their prediction embeddings at `slot`, reading the nodes
    * taken before it and themselves.
    */
  def answer(document: KVCache[Patch, V], taken: KVCache[Node, V], slot: Tensor0[Int32], asked: Tensor2[PoolQuery, Embedding, V]): Tensor2[PoolQuery, Embedding, V] =
    val slots = taken.keys.shape(Axis[Node])
    val normed = asked.vmap(Axis[PoolQuery])(selfAttentionPreNorm)
    val queries = Tensor1(Axis[PoolQuery]).fromRange(0 until asked.shape(Axis[PoolQuery]))
    val positions = queries *! slots +! slots +! slot
    decoded(
      document,
      asked,
      normed,
      KVCache(
        concatenate(taken.keys, normed.dot(Axis[Embedding])(selfWeights.keyWeights)),
        concatenate(taken.values, normed.dot(Axis[Embedding])(selfWeights.valueWeights))
      ),
      positions,
      concatenate(Tensor1(Axis[Node]).fromRange(0 until slots), positions),
      slots
    )

  /** The block applied to `rows` at `positions` of the joined sequence, attending to the `sources`
    * at `sourcePositions` and to the document.
    */
  private def decoded[Row: Λ, Source: Λ](
      document: KVCache[Patch, V],
      rows: Tensor2[Row, Embedding, V],
      normed: Tensor2[Row, Embedding, V],
      sources: KVCache[Source, V],
      positions: Tensor1[Row, Int32],
      sourcePositions: Tensor1[Source, Int32],
      slots: Int
  ): Tensor2[Row, Embedding, V] =
    val selfAttention = MultiHeadCustomAttention[Source, Embedding, Row, Embedding, V](selfWeights, jointSequenceMask[Row, Source](slots), AttentionScore.scaledDotProduct)
    val readSelf = selfAttention.attendAt(normed.dot(Axis[Embedding])(selfWeights.queryWeights), sources.keys, sources.values, positions, sourcePositions)
    val afterSelf = rows + selfAttention.projectHeads(readSelf)
    val documentRead = MultiHeadUnfusedFullAttention[Patch, PatchEmbedding, Row, Embedding, V](Axis[Patch], Axis[Row], params.documentAttention, AttentionScore.scaledDotProduct)
    val documentQueries = afterSelf.vmap(Axis[Row])(documentAttentionPreNorm).dot(Axis[Embedding])(params.documentAttention.queryWeights)
    val patches = Tensor1(Axis[Patch]).fromRange(0 until document.keys.shape(Axis[Patch]))
    val afterDocument = afterSelf + documentRead.projectHeads(documentRead.attendAt(documentQueries, document.keys, document.values, positions, patches))
    afterDocument + afterDocument.vmap(Axis[Row])(embedding => mlp(mlpPreNorm(embedding)))

/** Axis of the one node a decoding step takes. */
private trait Taking derives Label

object NodeDecoderBlock:

  case class Params[PatchEmbedding, Embedding, V](
      selfAttention: MultiHeadSelfAttention.Params[Embedding, V],
      selfAttentionNorm: LayerNorm.Params[Embedding, V],
      documentAttention: MultiHeadAttention.Params[PatchEmbedding, Embedding, V],
      documentAttentionNorm: LayerNorm.Params[Embedding, V],
      mlp: MLPEmbeddingMixer.Params[Embedding, V],
      mlpNorm: LayerNorm.Params[Embedding, V]
  )

  object Params:

    def xavierUniformDepthScaled[PatchEmbedding: Λ, Embedding: Λ, V: IsFloating](numBlocks: Int, numHeads: Int, patchEmbeddingExtent: AxisExtent[PatchEmbedding], embeddingExtent: AxisExtent[Embedding], embeddingMixedExtent: AxisExtent[EmbeddingMixed], key: Key, vtype: VType[V] = VType[Float32]): Params[PatchEmbedding, Embedding, V] =
      val (selfKey, documentKey, mlpKey) = key.splitToTuple(3)
      Params(
        selfAttention = MultiHeadSelfAttention.Params.xavierUniformDepthScaled(numBlocks, numHeads, embeddingExtent, selfKey, vtype),
        selfAttentionNorm = LayerNorm.Params.identity(embeddingExtent, vtype),
        documentAttention = MultiHeadAttention.Params.xavierUniformDepthScaled(numBlocks, numHeads, patchEmbeddingExtent, embeddingExtent, documentKey, vtype),
        documentAttentionNorm = LayerNorm.Params.identity(embeddingExtent, vtype),
        mlp = MLPEmbeddingMixer.Params.init(embeddingExtent, embeddingMixedExtent, mlpKey, vtype),
        mlpNorm = LayerNorm.Params.identity(embeddingExtent, vtype)
      )

/** Attention mask for a joined sequence of `slots` "taken" embeddings, then one "prediction" per
  * slot per query, each placed at its position in that sequence: the taken embedding of a slot at
  * the slot, the prediction of query `q` for slot `s` at `slots + q * slots + s`.
  * Rows read, columns are read:
  * {{{
  *                 source:  taken                 prediction
  *   target: taken          up to its own slot    nothing
  *   target: prediction     before its own slot   itself
  * }}}
  */
def jointSequenceMask[Target: Λ, Source: Λ](slots: Int): AttentionMask[Target, Source] = (targets, sources) =>
  targets.vmap(Axis[Target]): target =>
    val lastTakenRead = where(target < slots, target, (target - slots) % slots - 1)
    (sources elementEquals_! target) or (sources <=! lastTakenRead)
