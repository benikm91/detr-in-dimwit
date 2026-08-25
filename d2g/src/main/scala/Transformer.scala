import deepwit.activation.gelu
import deepwit.attention.AttentionScore
import deepwit.attention.MultiHeadAttention
import deepwit.attention.MultiHeadCausalSelfAttention
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
  * Training and transcription read the same weights through different doors — see
  * [[RecordDecoderBlock]] for which is which.
  */
class RecordDecoder[Patch: Λ, PatchEmbedding: Λ, Slot: Λ, Embedding: Λ, V: IsFloating](
    patchAxis: Axis[Patch],
    slotAxis: Axis[Slot],
    params: RecordDecoder.Params[PatchEmbedding, Embedding, V]
):

  private val blocks = params.blocks.map(block => RecordDecoderBlock(patchAxis, slotAxis, block))
  private val finalNorm = LayerNorm(params.finalNorm)

  /** The taken nodes as the decoder carries them, and what every position would answer with. */
  def forTraining(
      document: Tensor2[Patch, PatchEmbedding, V],
      taken: Tensor2[Slot, Embedding, V],
      remaining: Tensor2[Slot, Embedding, V]
  ): (Tensor2[Slot, Embedding, V], Tensor2[Slot, Embedding, V]) =
    val (carried, answered) = blocks.foldLeft((taken, remaining)):
      case ((taken, remaining), block) => block.forTraining(document, taken, remaining)
    (carried.vmap(slotAxis)(finalNorm), answered.vmap(slotAxis)(finalNorm))

  /** The nodes taken so far, read causally. */
  def forTranscription(document: Tensor2[Patch, PatchEmbedding, V], taken: Tensor2[Slot, Embedding, V]): Tensor2[Slot, Embedding, V] =
    blocks.foldLeft(taken)((decoded, block) => block.forTranscription(document, decoded)).vmap(slotAxis)(finalNorm)

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

/** The block a detection transformer decodes with — masked self-attention, then cross-attention
  * onto the encoded document, then the embedding mixer, each on its own residual branch.
  *
  * An embedding in a decoder has two jobs — become what its position predicts, and keep carrying
  * what its position holds for the others to read. Next-token prediction does both at once;
  * remaining-node prediction cannot, since the later slots have to know what is taken already in
  * order to answer with something else. So every slot gets both a *node embedding* carrying the
  * taken node and a *prediction embedding* becoming one of the remaining ones.
  *
  * The two doors differ only in what they read:
  *
  *   - [[forTraining]] takes both, joins them into one sequence twice as long and splits the
  *     answer again, so the layout and the [[RecordDecoderBlock.jointSequenceMask]] that goes with it never
  *     leave this class.
  *   - [[forTranscription]] takes the nodes taken so far and nothing beside them, read causally,
  *     which is the same rule with the prediction half absent.
  *
  * Neither takes a mask. That is the point: the mask has to be one shape in training and another
  * in transcription, and a caller handing one in cannot be seen to have got it right.
  */
class RecordDecoderBlock[Patch: Λ, PatchEmbedding: Λ, Slot: Λ, Embedding: Λ, V: IsFloating](
    patchAxis: Axis[Patch],
    params: RecordDecoderBlock.Params[PatchEmbedding, Embedding, V]
):

  import RecordDecoderBlock.{Node, Prediction, DecoderSlot}

  private val slotAxis = Axis[DecoderSlot]

  private val jointAttention = MultiHeadCustomSelfAttention(
    params.selfAttention,
    RecordDecoderBlock.jointSequenceMask,
    AttentionScore.scaledDotProduct
  )
  private val causalAttention = MultiHeadCausalSelfAttention(slotAxis, params.selfAttention)
  private val selfAttentionPreNorm = LayerNorm(params.selfAttentionNorm)
  private val crossAttention = MultiHeadFullAttention(patchAxis, slotAxis, params.crossAttention)
  private val crossAttentionPreNorm = LayerNorm(params.crossAttentionNorm)
  private val mlp = MLPEmbeddingMixer(params.mlp)
  private val mlpPreNorm = LayerNorm(params.mlpNorm)

  /** The taken nodes and the prediction embeddings beside them, one slot each. */
  def forTraining(
      document: Tensor2[Patch, PatchEmbedding, V],
      nodes: Tensor2[Node, Embedding, V],
      predictions: Tensor2[Prediction, Embedding, V]
  ): (Tensor2[Node, Embedding, V], Tensor2[Prediction, Embedding, V]) =
    val slots = nodes.shape(Axis[Node])
    var x = concatenate(nodes, predictions)
    x = x + jointAttention(x.vmap(slotAxis)(selfAttentionPreNorm)) // self attention
    x = x + crossAttention(document, x.vmap(slotAxis)(crossAttentionPreNorm)) // cross attention
    x = x + x.vmap(slotAxis)(embedding => mlp(mlpPreNorm(embedding))) // embedding mixer
    val numNodes = x.shape(slotAxis) / 2
    val numPredictions = x.shape(slotAxis) - numNodes
    x.deconcatenate(slotAxis, (Axis[Node] -> numNodes, Axis[Prediction] -> numPredictions))

  /** The nodes taken so far, read causally. */
  def forTranscription(document: Tensor2[Patch, PatchEmbedding, V], taken: Tensor2[Node, Embedding, V], prediction: Tensor1[Embedding, V]): Tensor1[Embedding, V] =
    var x = concatenate(taken, prediction.prependAxis(Axis[Prediction]))
    x = x + causalAttention(x.vmap(slotAxis)(selfAttentionPreNorm)) // self attention
    x = x + crossAttention(document, x.vmap(slotAxis)(crossAttentionPreNorm)) // cross attention
    x = x + x.vmap(slotAxis)(embedding => mlp(mlpPreNorm(embedding))) // embedding mixer
    x.slice(slotAxis.at(-1)) // last position is (mixed) prediction vector

object RecordDecoderBlock:

  /** Which of the `2 * slots` embeddings of a joined training sequence each of them may attend to.
    *
    * The sequence is the node embeddings of the record taken so far followed by the prediction
    * embeddings beside them, so the mask falls into four blocks of one slot each — a row is an
    * embedding that reads, a column one that is read:
    *
    * {{{
    *                          node (source)         prediction (source)
    *   node (target)          up to its own slot    nothing
    *   prediction (target)    before its own slot   itself
    * }}}
    *
    * A node embedding carries the record as far as itself; a prediction embedding reads exactly
    * what is taken before its own slot, so that what it may answer with is what is left over; and
    * nothing reads a prediction embedding, which holds a guess rather than a node. The diagonal of
    * the last block is what keeps the first prediction row, which has nothing taken before it,
    * from being fully masked — a row of nothing but `-inf` has no softmax.
    */
  def jointSequenceMask(maskExtent: AxisExtent[DecoderSlot]): Tensor2[DecoderSlot, DecoderSlot, Bool] =

    // Define vocabulary to clarify mask creation
    trait NodeSource derives Label
    trait NodeTarget derives Label
    trait PredictionSource derives Label
    trait PredictionTarget derives Label

    val blockSize = maskExtent.size / 2
    def blockShape[S1: Label, S2: Label]: Shape2[S1, S2] =
      Shape2(Axis[S1] -> blockSize, Axis[S2] -> blockSize)

    val upToItsOwnSlot = tril(Tensor(blockShape[NodeTarget, NodeSource]).fill(true))
    val noSlot = Tensor(blockShape[NodeTarget, PredictionSource]).fill(false)

    val beforeItsOwnSlot = tril(Tensor(blockShape[PredictionTarget, NodeSource]).fill(true), kthDiagonal = -1)
    val itselfOnly = Tensor2.eye(blockShape[PredictionTarget, PredictionSource], VType[Bool])

    val mask = concatenate(
      concatenate(upToItsOwnSlot, noSlot),
      concatenate(beforeItsOwnSlot, itselfOnly)
    )
    // drop internal vocabulary
    mask.relabelAll((Axis[DecoderSlot], Axis[DecoderSlot]))

  type DecoderSlot = Node |+| Prediction

  trait Node derives Label
  trait Prediction derives Label

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
