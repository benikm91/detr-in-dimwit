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

trait Query derives Label // Queries of the underlying query pool.

/** Which embeddings of a joined training sequence each of them may attend to.
  *
  * The sequence is the embeddings of what is taken so far followed by one prediction embedding per
  * slot per query asked — a pair of queries while training, the whole pool while transcribing — so
  * the mask falls into four blocks: a row is an embedding that reads, a column one that is read:
  *
  * {{{
  *                 source:  taken                 prediction
  *   target: taken          up to its own slot    nothing
  *   target: prediction     before its own slot   itself
  * }}}
  *
  * A taken embedding carries the record as far as itself; a prediction embedding reads exactly what
  * is taken before the slot it answers for, so that what it may answer with is what is left over;
  * and nothing reads a prediction embedding, which holds a guess rather than a record.
  *
  * The last block is the diagonal and stays the diagonal however many tokens a slot has: the tokens
  * of one slot must not read each other, or they would agree on an answer between themselves rather
  * than each having to find one. It is also what keeps the first prediction row, which has nothing
  * taken before it, from being fully masked — a row of nothing but `-inf` has no softmax.
  */
def jointSequenceMask[Context: Λ](queries: Int)(context: AxisExtent[Context]): Tensor2[Context, Context, Bool] =

  trait TakenSource derives Label
  trait TakenTarget derives Label
  trait PredictionSource derives Label
  trait PredictionTarget derives Label

  val slots = context.size / (1 + queries)
  val predictions = slots * queries
  val taken = Axis[TakenSource] -> slots

  val upToItsOwnSlot = tril(Tensor(Shape2(Axis[TakenTarget] -> slots, taken)).fill(true))
  val noSlot = Tensor(Shape2(Axis[TakenTarget] -> slots, Axis[PredictionSource] -> predictions)).fill(false)

  // A prediction row answers for the slot it sits at, which is its position within its own token's
  // block, so the taken embeddings it may read are the ones before that slot.
  val answersFor = Tensor1(Axis[PredictionTarget], VType[Int32])
    .fromArray(Array.range(0, predictions).map(_ % slots))
  val readable = Tensor1(taken.axis, VType[Int32]).fromArray(Array.range(0, slots))
  val shape = Shape2(Axis[PredictionTarget] -> predictions, taken)
  val beforeItsOwnSlot = readable.broadcastTo(shape) < answersFor.broadcastTo(shape)

  val itselfOnly = Tensor2(Axis[PredictionTarget] -> predictions, Axis[PredictionSource] -> predictions).eye(VType[Bool])

  val mask = concatenate(
    concatenate(upToItsOwnSlot, noSlot),
    concatenate(beforeItsOwnSlot, itselfOnly)
  )
  mask.relabelAll((Axis[Context], Axis[Context]))

/** Axis of the node prediction embeddings a decoder attends over: every query's answer for every
  * node slot, laid out one whole block of slots per query. It is [[Query]] and [[Node]] flattened,
  * because attention reads one sequence — outside a decoder the two are separate axes.
  */
trait NodePrediction derives Label

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
      asked: Tensor3[Query, Node, Embedding, V]
  ): (Tensor2[Node, Embedding, V], Tensor3[Query, Node, Embedding, V]) =
    val perQuery = Shape2(asked.shape.extent(Axis[Query]), asked.shape.extent(Axis[Node]))
    val predictions = asked.flatten((Axis[Query], Axis[Node])).relabelAll((Axis[NodePrediction], Axis[Embedding]))
    val (carried, answered) = blocks.foldLeft((nodes, predictions)):
      case ((nodes, predictions), block) => block.forTraining(document, nodes, predictions)
    (
      carried.vmap(Axis[Node])(finalNorm),
      answered.vmap(Axis[NodePrediction])(finalNorm).unflatten(Axis[NodePrediction], perQuery)
    )

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

  private def jointAttention(queries: Int) = MultiHeadCustomSelfAttention(
    params.selfAttention,
    jointSequenceMask[Context](queries),
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
    val queries = predictions.shape(Axis[NodePrediction]) / nodes.shape(Axis[Node])
    var x = concatenate(nodes, predictions)
    x = x + jointAttention(queries)(x.vmap(contextAxis)(selfAttentionPreNorm))
    x = x + documentAttention(document, x.vmap(contextAxis)(documentAttentionPreNorm))
    x = x + x.vmap(contextAxis)(embedding => mlp(mlpPreNorm(embedding)))
    x.deconcatenate(contextAxis, (nodes.extent(Axis[Node]), predictions.extent(Axis[NodePrediction])))

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

/** The same for the relationships: [[Query]] and [[Edge]] flattened, one block of slots per query. */
trait EdgePrediction derives Label

/** What an [[EdgeDecoder]] attends onto: the nodes as a [[NodeDecoder]] carries them, and which of
  * the slots hold a node at all. The positions past a record are there to make every record the
  * same shape and relate nothing, so the two always travel together.
  */
case class NodeSource[Embedding, V](nodeEmbeddings: Tensor2[Node, Embedding, V], presentMask: Tensor1[Node, Bool])

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
      asked: Tensor3[Query, Edge, Embedding, V]
  ): (Tensor2[Edge, Embedding, V], Tensor3[Query, Edge, Embedding, V]) =
    val perQuery = Shape2(asked.shape.extent(Axis[Query]), asked.shape.extent(Axis[Edge]))
    val predictions = asked.flatten((Axis[Query], Axis[Edge])).relabelAll((Axis[EdgePrediction], Axis[Embedding]))
    val (carried, answered) = blocks.foldLeft((edges, predictions)):
      case ((edges, predictions), block) => block.forTraining(document, nodes, edges, predictions)
    (
      carried.vmap(Axis[Edge])(finalNorm),
      answered.vmap(Axis[EdgePrediction])(finalNorm).unflatten(Axis[EdgePrediction], perQuery)
    )

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

  private def jointAttention(queries: Int) = MultiHeadCustomSelfAttention(
    params.selfAttention,
    jointSequenceMask[Context](queries),
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
    val queries = predictions.shape(Axis[EdgePrediction]) / edges.shape(Axis[Edge])
    var x = concatenate(edges, predictions)
    x = x + jointAttention(queries)(x.vmap(contextAxis)(selfAttentionPreNorm))
    x = x + documentAttention(document, x.vmap(contextAxis)(documentAttentionPreNorm))
    x = x + nodeAttention(nodes.presentMask, x.shape.extent(contextAxis))(nodes.nodeEmbeddings, x.vmap(contextAxis)(nodeAttentionPreNorm))
    x = x + x.vmap(contextAxis)(embedding => mlp(mlpPreNorm(embedding)))
    x.deconcatenate(contextAxis, (edges.extent(Axis[Edge]), predictions.extent(Axis[EdgePrediction])))

  /** Attention onto the nodes that are there, which is data rather than shape and so is built
    * around the mask it is given rather than once.
    *
    * Where a record holds no node at all every slot is read instead, since a row of nothing but
    * `-inf` has no softmax.
    */
  private def nodeAttention(presentMask: Tensor1[Node, Bool], context: AxisExtent[Context]) =
    val anyNode = presentMask.any.broadcastTo(presentMask.shape)
    val readable = where(anyNode, presentMask, Tensor.like(presentMask).fill(true))
    val mask = readable.broadcastTo(Shape2(context, presentMask.shape.extent(Axis[Node])))
    MultiHeadCustomAttention[Node, Embedding, Context, Embedding, V](params.nodeAttention, _ => mask, AttentionScore.scaledDotProduct)

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
