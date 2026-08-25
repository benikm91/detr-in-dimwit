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

/** The decoder of the record, read through two doors.
  *
  * An embedding in a decoder has two jobs — become what its slot predicts, and keep carrying what
  * its slot holds for the others to read. Remaining-node prediction cannot do both at once, since
  * a later slot has to know what is taken already in order to answer with something else. So every
  * slot gets both: a *node embedding* carrying the taken node, and a *prediction embedding*
  * becoming one of the remaining ones.
  *
  * Neither door takes a mask. What a slot may read is not the caller's to say.
  */
class RecordDecoder[PatchEmbedding: Λ, Embedding: Λ, V: IsFloating](
    params: RecordDecoder.Params[PatchEmbedding, Embedding, V]
):

  import RecordDecoder.Prediction

  private val blocks = params.blocks.map(RecordDecoderBlock(_))
  private val finalNorm = LayerNorm(params.finalNorm)

  /** The taken nodes as the decoder carries them, and what every slot would answer with. */
  def forTraining(
      document: Tensor2[Patch, PatchEmbedding, V],
      nodes: Tensor2[Node, Embedding, V],
      predictions: Tensor2[Prediction, Embedding, V]
  ): (Tensor2[Node, Embedding, V], Tensor2[Prediction, Embedding, V]) =
    val (carried, answered) = blocks.foldLeft((nodes, predictions)):
      case ((nodes, predictions), block) => block.forTraining(document, nodes, predictions)
    (carried.vmap(Axis[Node])(finalNorm), answered.vmap(Axis[Prediction])(finalNorm))

  /** What the decoder would answer with next, after the nodes taken so far. */
  def forTranscription(
      document: Tensor2[Patch, PatchEmbedding, V],
      taken: Tensor2[Node, Embedding, V],
      prediction: Tensor1[Embedding, V]
  ): Tensor1[Embedding, V] =
    val (_, answered) = blocks.foldLeft((taken, prediction)):
      case ((taken, prediction), block) => block.forTranscription(document, taken, prediction)
    finalNorm(answered)

object RecordDecoder:

  /** Axis of the prediction embeddings, one beside every node slot: what the decoder would answer
    * with there, rather than what is taken there.
    */
  trait Prediction derives Label

  /** The sequence a block decodes: every node embedding, then every prediction embedding. */
  type DecoderContext = Node |+| Prediction

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

/** One block of the [[RecordDecoder]]: masked self-attention, then cross-attention onto the
  * encoded document, then the embedding mixer, each on its own residual branch.
  *
  * The two doors join the halves they are given into one sequence and take the answer apart again,
  * so how they are laid out, and the mask that goes with the layout, stay in this class.
  */
class RecordDecoderBlock[PatchEmbedding: Λ, Embedding: Λ, V: IsFloating](
    params: RecordDecoderBlock.Params[PatchEmbedding, Embedding, V]
):

  import RecordDecoder.{Prediction, DecoderContext}

  private val contextAxis = Axis[DecoderContext]

  private val jointAttention = MultiHeadCustomSelfAttention(
    params.selfAttention,
    RecordDecoderBlock.jointSequenceMask,
    AttentionScore.scaledDotProduct
  )
  private val causalAttention = MultiHeadCausalSelfAttention(contextAxis, params.selfAttention)
  private val selfAttentionPreNorm = LayerNorm(params.selfAttentionNorm)
  private val crossAttention = MultiHeadFullAttention(Axis[Patch], contextAxis, params.crossAttention)
  private val crossAttentionPreNorm = LayerNorm(params.crossAttentionNorm)
  private val mlp = MLPEmbeddingMixer(params.mlp)
  private val mlpPreNorm = LayerNorm(params.mlpNorm)

  def forTraining(
      document: Tensor2[Patch, PatchEmbedding, V],
      nodes: Tensor2[Node, Embedding, V],
      predictions: Tensor2[Prediction, Embedding, V]
  ): (Tensor2[Node, Embedding, V], Tensor2[Prediction, Embedding, V]) =
    var x = concatenate(nodes, predictions)
    x = x + jointAttention(x.vmap(contextAxis)(selfAttentionPreNorm)) // self attention
    x = x + crossAttention(document, x.vmap(contextAxis)(crossAttentionPreNorm)) // cross attention
    x = x + x.vmap(contextAxis)(embedding => mlp(mlpPreNorm(embedding))) // embedding mixer
    x.deconcatenate(contextAxis, (nodes.extent(Axis[Node]), predictions.extent(Axis[Prediction])))

  def forTranscription(
      document: Tensor2[Patch, PatchEmbedding, V],
      taken: Tensor2[Node, Embedding, V],
      prediction: Tensor1[Embedding, V]
  ): (Tensor2[Node, Embedding, V], Tensor1[Embedding, V]) =
    var x = concatenate(taken, prediction.prependAxis(Axis[Prediction]))
    x = x + causalAttention(x.vmap(contextAxis)(selfAttentionPreNorm)) // self attention
    x = x + crossAttention(document, x.vmap(contextAxis)(crossAttentionPreNorm)) // cross attention
    x = x + x.vmap(contextAxis)(embedding => mlp(mlpPreNorm(embedding))) // embedding mixer
    val (carried, answered) = x.deconcatenate(contextAxis, (taken.extent(Axis[Node]), Axis[Prediction] -> 1))
    (carried, answered.squeeze(Axis[Prediction]))

object RecordDecoderBlock:

  import RecordDecoder.DecoderContext

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
  def jointSequenceMask(maskExtent: AxisExtent[DecoderContext]): Tensor2[DecoderContext, DecoderContext, Bool] =

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
    mask.relabelAll((Axis[DecoderContext], Axis[DecoderContext]))

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
        mlp = MLPEmbeddingMixer.Params.init(embeddingExtent, embeddingMixedExtent, mlpKey, vtype),
        mlpNorm = LayerNorm.Params.identity(embeddingExtent, vtype)
      )
