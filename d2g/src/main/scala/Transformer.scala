import deepwit.activation.gelu
import deepwit.attention.AttentionScore
import deepwit.attention.MultiHeadAttention
import deepwit.attention.MultiHeadCausalSelfAttention
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

    def init[Embedding: Λ, V: IsFloating](embeddingExtent: AxisExtent[Embedding], embeddingMixedExtent: AxisExtent[EmbeddingMixed], key: Key, vtype: VType[V] = VType[Float32]): Params[Embedding, V] =
      val (expandKey, projectKey) = key.splitToTuple(2)
      Params(
        expand = AffineLayer.Params.init(embeddingExtent, embeddingMixedExtent, expandKey, vtype),
        project = AffineLayer.Params.init(embeddingMixedExtent, embeddingExtent, projectKey, vtype)
      )

/** A full self-attention encoder of the document's patches. */
class DocumentEncoder[Embedding: Λ, V: IsFloating](
    params: DocumentEncoder.Params[Embedding, V]
) extends (Tensor2[Patch, Embedding, V] => Tensor2[Patch, Embedding, V]):

  private val blocks = params.blocks.map(DocumentEncoderBlock(_))
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

class DocumentEncoderBlock[Embedding: Λ, V: IsFloating](
    params: DocumentEncoderBlock.Params[Embedding, V]
) extends TransformerBlock[Patch, Embedding, V](Axis[Patch]):

  private val selfAttention = MultiHeadFullSelfAttention(Axis[Patch], params.selfAttention)
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
        mlp = MLPEmbeddingMixer.Params.init(embeddingExtent, embeddingMixedExtent, mlpKey, vtype),
        mlpNorm = LayerNorm.Params.identity(embeddingExtent, vtype)
      )

/** Which of the `2 * slots` embeddings of a joined training sequence each of them may attend to.
  *
  * The sequence is the embeddings of what is taken so far followed by the prediction embeddings
  * beside them, so the mask falls into four blocks of one slot each — a row is an embedding that
  * reads, a column one that is read:
  *
  * {{{
  *                 source:  taken                 prediction
  *   target: taken          up to its own slot    nothing
  *   target: prediction     before its own slot   itself
  * }}}
  *
  * A taken embedding carries the record as far as itself; a prediction embedding reads exactly
  * what is taken before its own slot, so that what it may answer with is what is left over; and
  * nothing reads a prediction embedding, which holds a guess rather than a record. The diagonal of
  * the last block is what keeps the first prediction row, which has nothing taken before it, from
  * being fully masked — a row of nothing but `-inf` has no softmax.
  */
def jointSequenceMask[Context: Λ](context: AxisExtent[Context]): Tensor2[Context, Context, Bool] =

  // Define vocabulary to clarify mask creation
  trait TakenSource derives Label
  trait TakenTarget derives Label
  trait PredictionSource derives Label
  trait PredictionTarget derives Label

  val blockSize = context.size / 2
  def blockShape[S1: Label, S2: Label]: Shape2[S1, S2] =
    Shape2(Axis[S1] -> blockSize, Axis[S2] -> blockSize)

  val upToItsOwnSlot = tril(Tensor(blockShape[TakenTarget, TakenSource]).fill(true))
  val noSlot = Tensor(blockShape[TakenTarget, PredictionSource]).fill(false)

  val beforeItsOwnSlot = tril(Tensor(blockShape[PredictionTarget, TakenSource]).fill(true), kthDiagonal = -1)
  val itselfOnly = Tensor2.eye(blockShape[PredictionTarget, PredictionSource], VType[Bool])

  val mask = concatenate(
    concatenate(upToItsOwnSlot, noSlot),
    concatenate(beforeItsOwnSlot, itselfOnly)
  )
  // drop internal vocabulary
  mask.relabelAll((Axis[Context], Axis[Context]))

/** Axis of the node prediction embeddings, one beside every node slot: what the decoder would
  * answer with there, rather than what is taken there.
  */
trait NodePrediction derives Label

/** The decoder of the record's nodes, read through two doors.
  *
  * An embedding in a decoder has two jobs — become what its slot predicts, and keep carrying what
  * its slot holds for the others to read. Remaining-node prediction cannot do both at once, since
  * a later slot has to know what is taken already in order to answer with something else. So every
  * slot gets both: a *node embedding* carrying the taken node, and a *prediction embedding*
  * becoming one of the remaining ones.
  *
  * Neither door takes a mask. What a slot may read is not the caller's to say.
  */
class NodeDecoder[PatchEmbedding: Λ, Embedding: Λ, V: IsFloating](
    params: NodeDecoder.Params[PatchEmbedding, Embedding, V]
):

  private val blocks = params.blocks.map(NodeDecoderBlock(_))
  private val finalNorm = LayerNorm(params.finalNorm)

  /** The taken nodes as the decoder carries them, and what every slot would answer with. */
  def forTraining(
      document: Tensor2[Patch, PatchEmbedding, V],
      nodes: Tensor2[Node, Embedding, V],
      predictions: Tensor2[NodePrediction, Embedding, V]
  ): (Tensor2[Node, Embedding, V], Tensor2[NodePrediction, Embedding, V]) =
    val (carried, answered) = blocks.foldLeft((nodes, predictions)):
      case ((nodes, predictions), block) => block.forTraining(document, nodes, predictions)
    (carried.vmap(Axis[Node])(finalNorm), answered.vmap(Axis[NodePrediction])(finalNorm))

  /** The nodes taken so far as the decoder carries them — which is what an [[EdgeDecoder]]
    * relates — and what it would answer with next.
    */
  def forTranscription(
      document: Tensor2[Patch, PatchEmbedding, V],
      taken: Tensor2[Node, Embedding, V],
      prediction: Tensor1[Embedding, V]
  ): (Tensor2[Node, Embedding, V], Tensor1[Embedding, V]) =
    val (carried, answered) = blocks.foldLeft((taken, prediction)):
      case ((taken, prediction), block) => block.forTranscription(document, taken, prediction)
    (carried.vmap(Axis[Node])(finalNorm), finalNorm(answered))

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
  * The two doors join the halves they are given into one sequence and take the answer apart again,
  * so how they are laid out, and the mask that goes with the layout, stay in this class.
  */
class NodeDecoderBlock[PatchEmbedding: Λ, Embedding: Λ, V: IsFloating](
    params: NodeDecoderBlock.Params[PatchEmbedding, Embedding, V]
):

  import NodeDecoder.Context

  private val contextAxis = Axis[Context]

  private val jointAttention = MultiHeadCustomSelfAttention(
    params.selfAttention,
    jointSequenceMask[Context],
    AttentionScore.scaledDotProduct
  )
  private val causalAttention = MultiHeadCausalSelfAttention(contextAxis, params.selfAttention)
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
    x = x + jointAttention(x.vmap(contextAxis)(selfAttentionPreNorm)) // self attention
    x = x + documentAttention(document, x.vmap(contextAxis)(documentAttentionPreNorm)) // cross attention on the document
    x = x + x.vmap(contextAxis)(embedding => mlp(mlpPreNorm(embedding))) // embedding mixer
    x.deconcatenate(contextAxis, (nodes.extent(Axis[Node]), predictions.extent(Axis[NodePrediction])))

  def forTranscription(
      document: Tensor2[Patch, PatchEmbedding, V],
      taken: Tensor2[Node, Embedding, V],
      prediction: Tensor1[Embedding, V]
  ): (Tensor2[Node, Embedding, V], Tensor1[Embedding, V]) =
    var x = concatenate(taken, prediction.prependAxis(Axis[NodePrediction]))
    x = x + causalAttention(x.vmap(contextAxis)(selfAttentionPreNorm)) // self attention
    x = x + documentAttention(document, x.vmap(contextAxis)(documentAttentionPreNorm)) // cross attention on the document
    x = x + x.vmap(contextAxis)(embedding => mlp(mlpPreNorm(embedding))) // embedding mixer
    val (carried, answered) = x.deconcatenate(contextAxis, (taken.extent(Axis[Node]), Axis[NodePrediction] -> 1))
    (carried, answered.squeeze(Axis[NodePrediction]))

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

/** Axis of the relationship prediction embeddings, one beside every relationship slot. */
trait EdgePrediction derives Label

/** The decoder of the record's relationships, read through the same two doors as the
  * [[NodeDecoder]].
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

  /** The taken relationships as the decoder carries them, and what every slot would answer with.
    *
    * `holdsNode` says which of the node slots hold a node, since the positions a record does not
    * reach are there to make every record the same shape and relate nothing.
    */
  def forTraining(
      document: Tensor2[Patch, PatchEmbedding, V],
      nodes: Tensor2[Node, Embedding, V],
      holdsNode: Tensor1[Node, Bool],
      edges: Tensor2[Edge, Embedding, V],
      predictions: Tensor2[EdgePrediction, Embedding, V]
  ): (Tensor2[Edge, Embedding, V], Tensor2[EdgePrediction, Embedding, V]) =
    val (carried, answered) = blocks.foldLeft((edges, predictions)):
      case ((edges, predictions), block) => block.forTraining(document, nodes, holdsNode, edges, predictions)
    (carried.vmap(Axis[Edge])(finalNorm), answered.vmap(Axis[EdgePrediction])(finalNorm))

  /** What the decoder would answer with next, after the relationships taken so far. */
  def forTranscription(
      document: Tensor2[Patch, PatchEmbedding, V],
      nodes: Tensor2[Node, Embedding, V],
      holdsNode: Tensor1[Node, Bool],
      taken: Tensor2[Edge, Embedding, V],
      prediction: Tensor1[Embedding, V]
  ): Tensor1[Embedding, V] =
    val (_, answered) = blocks.foldLeft((taken, prediction)):
      case ((taken, prediction), block) => block.forTranscription(document, nodes, holdsNode, taken, prediction)
    finalNorm(answered)

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

  private val jointAttention = MultiHeadCustomSelfAttention(
    params.selfAttention,
    jointSequenceMask[Context],
    AttentionScore.scaledDotProduct
  )
  private val causalAttention = MultiHeadCausalSelfAttention(contextAxis, params.selfAttention)
  private val selfAttentionPreNorm = LayerNorm(params.selfAttentionNorm)
  private val documentAttention = MultiHeadFullAttention(Axis[Patch], contextAxis, params.documentAttention)
  private val documentAttentionPreNorm = LayerNorm(params.documentAttentionNorm)
  private val nodeAttentionPreNorm = LayerNorm(params.nodeAttentionNorm)
  private val mlp = MLPEmbeddingMixer(params.mlp)
  private val mlpPreNorm = LayerNorm(params.mlpNorm)

  def forTraining(
      document: Tensor2[Patch, PatchEmbedding, V],
      nodes: Tensor2[Node, Embedding, V],
      holdsNode: Tensor1[Node, Bool],
      edges: Tensor2[Edge, Embedding, V],
      predictions: Tensor2[EdgePrediction, Embedding, V]
  ): (Tensor2[Edge, Embedding, V], Tensor2[EdgePrediction, Embedding, V]) =
    var x = concatenate(edges, predictions)
    x = x + jointAttention(x.vmap(contextAxis)(selfAttentionPreNorm)) // self attention
    x = x + documentAttention(document, x.vmap(contextAxis)(documentAttentionPreNorm)) // cross attention on the document
    x = x + nodeAttention(holdsNode, x.shape.extent(contextAxis))(nodes, x.vmap(contextAxis)(nodeAttentionPreNorm)) // cross attention on the nodes that are there
    x = x + x.vmap(contextAxis)(embedding => mlp(mlpPreNorm(embedding))) // embedding mixer
    x.deconcatenate(contextAxis, (edges.extent(Axis[Edge]), predictions.extent(Axis[EdgePrediction])))

  def forTranscription(
      document: Tensor2[Patch, PatchEmbedding, V],
      nodes: Tensor2[Node, Embedding, V],
      holdsNode: Tensor1[Node, Bool],
      taken: Tensor2[Edge, Embedding, V],
      prediction: Tensor1[Embedding, V]
  ): (Tensor2[Edge, Embedding, V], Tensor1[Embedding, V]) =
    var x = concatenate(taken, prediction.prependAxis(Axis[EdgePrediction]))
    x = x + causalAttention(x.vmap(contextAxis)(selfAttentionPreNorm)) // self attention
    x = x + documentAttention(document, x.vmap(contextAxis)(documentAttentionPreNorm)) // cross attention on the document
    x = x + nodeAttention(holdsNode, x.shape.extent(contextAxis))(nodes, x.vmap(contextAxis)(nodeAttentionPreNorm)) // cross attention on the nodes that are there
    x = x + x.vmap(contextAxis)(embedding => mlp(mlpPreNorm(embedding))) // embedding mixer
    val (carried, answered) = x.deconcatenate(contextAxis, (taken.extent(Axis[Edge]), Axis[EdgePrediction] -> 1))
    (carried, answered.squeeze(Axis[EdgePrediction]))

  /** Attention onto the nodes that are there, which is data rather than shape and so is built
    * around the mask it is given rather than once.
    *
    * Where a record holds no node at all every slot is read instead, since a row of nothing but
    * `-inf` has no softmax.
    */
  private def nodeAttention(holdsNode: Tensor1[Node, Bool], context: AxisExtent[Context]) =
    val readable = where_!(holdsNode.any, holdsNode, Tensor.like(holdsNode).fill(true))
    val mask = readable.broadcastTo(Shape2(context, holdsNode.shape.extent(Axis[Node])))
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
