package d2g.model

import d2g.*
import d2g.train.*
import d2g.eval.*
import d2g.config.*
import d2s.model.*
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
import dimwit.Label as Λ

import scala.language.implicitConversions

/** Every [[PoolQuery]]'s answer for every [[Edge]] slot. */
type EdgePrediction = PoolQuery |*| Edge

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

  private def nodeAttention(presentMask: Tensor1[Node, Bool], context: AxisExtent[Context]) =
    val readable = where_!(presentMask.any, presentMask, true) // in case no nodes are present => make all slots readable to prevent NaNs in attention
    val mask = readable.broadcastTo(Shape2(context, readable.shape.extent(Axis[Node])))
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
