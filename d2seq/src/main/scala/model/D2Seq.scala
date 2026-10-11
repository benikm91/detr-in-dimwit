package d2seq.model

import d2seq.*
import dataset.IsConstruction
import dataset.NodeClass
import dataset.RecordNodes
import deepwit.attention.KVCache
import deepwit.base.AffineLayer
import deepwit.embedder.LearnedAbsolutePositionalInjector
import deepwit.embedder.VocabularyEmbedder
import deepwit.init.Init
import documentEncoder.DocumentEncoder
import NodeHead.NodeLogits
import dimwit.*

import scala.language.implicitConversions

/** Document-to-sequence model based on next-node prediction.
  *
  * 1. The document (or: image) is embedded by a vision transformer to a sequence of patch embeddings.
  * 2. The nodes are written one after the other, each from one embedding that reads the document
  *    (1) and the nodes before it: the first from a learned start, every later one from the node
  *    before it.
  */
class D2Seq[V: IsFloating](params: D2Seq.Params[V]):

  val encodeDocument = DocumentEncoder(params.encoder)

  private val embedNodes = NodeEmbedder(params.embedder)
  private val nodePosition = LearnedAbsolutePositionalInjector(params.positions)
  private val decoder = SequenceDecoder(params.decoder)
  val nodeHead = NodeHead(params.head)

  /** What every slot answers, read from the nodes of `written` before it: the slot past the last
    * node answers whether the record ends there.
    */
  def logits(document: Tensor3[Width, Height, Channel, V], written: RecordNodes[Node]): NodeLogits[V] =
    val slots = written.nodeClass.shape(Axis[Node])
    val before = embedNodes(written).slice(Axis[Node].at(0 until slots - 1))
    val read = concatenate(params.start.prependAxis(Axis[Node]), before, Axis[Node])
    nodeHead(decoder(encodeDocument(document), nodePosition(read)))

  // Writing a record down one node at a time, as [[logits]] would have it at every slot, but
  // projecting at each step only what the step adds.

  /** What the first slot reads. */
  val start: Tensor1[Embedding, V] = params.start

  /** The encoded document as every step of the decoder reads it. */
  def read(encodedDocument: Tensor2[Patch, Embedding, V]): List[KVCache[Patch, V]] =
    decoder.read(encodedDocument)

  /** The decoder before any node is written. */
  def nothingWritten(slots: AxisExtent[Node]): List[KVCache[Node, V]] =
    decoder.nothingWritten(slots)

  /** What the slot after `node`, a record of one node, reads. */
  def embed(node: RecordNodes[Node]): Tensor1[Embedding, V] =
    embedNodes(node).squeeze(Axis[Node])

  /** What `slot` answers, reading `previous` — [[start]] at the first slot, the [[embed]]ded node
    * before it after that — and the nodes `written` before it: logits of one node, and `written`
    * with the slot among them.
    */
  def answerAt(documentCache: List[KVCache[Patch, V]], written: List[KVCache[Node, V]], slot: Tensor0[Int32], previous: Tensor1[Embedding, V]): (NodeLogits[V], List[KVCache[Node, V]]) =
    val (answered, nowWritten) = decoder.decode(documentCache, written, slot, nodePosition.injectAt(slot)(previous))
    (nodeHead(answered.prependAxis(Axis[Node])), nowWritten)

object D2Seq:

  case class Params[V](
      encoder: DocumentEncoder.Params[Embedding, V],
      decoder: SequenceDecoder.Params[Embedding, Embedding, V],
      embedder: NodeEmbedder.Params[V],
      head: NodeHead.Params[V],
      start: Tensor1[Embedding, V],
      positions: LearnedAbsolutePositionalInjector.Params[Node, Embedding, V]
  )

  object Params:

    given tensorTree: TensorTree[Params[Float32]] = TensorTree.derived
    given tree: TreeOf[Params[Float32], Float32] = TreeOf.derived

    /** @param nodes How many nodes of a record the model can hold. One more than the most any record
      *              of the data draws, so that the slot past the last node has somewhere to say the
      *              nodes have ended.
      * @param canvas The width of the drawing, which is how many pixels a coordinate chooses from.
      */
    def init(
        numLayers: Int,
        numHeads: Int,
        embedding: Int,
        nodes: Int,
        canvas: Int,
        key: Key
    ): Params[Float32] =
      val (encoderKey, decoderKey, embedderKey, headKey, startKey, positionKey) = key.splitToTuple(6)

      val embeddingExtent = Axis[Embedding] -> embedding
      val embeddingMixedExtent = Axis[DocumentEncoder.EmbeddingMixed] -> embedding * 4
      val partExtent = Axis[PartEmbedding] -> embedding / 8
      val nodeClassExtent = Axis[dataset.NodeClasses] -> NodeClass.values.length
      val constructionExtent = Axis[IsConstruction] -> 2
      val pixelExtent = Axis[Pixel] -> canvas
      // A node embedding is put together from its class, whether it is construction geometry, and
      // the six coordinates a class can place.
      val nodePartExtent = Axis[NodePart |*| PartEmbedding] -> 8 * partExtent.size

      val (startXKey, startYKey, endXKey, endYKey, midXKey, midYKey, nodeClassKey, constructionKey, nodeProjectionKey) = embedderKey.splitToTuple(9)
      val (classHeadKey, constructionHeadKey, startXHeadKey, startYHeadKey, endXHeadKey, endYHeadKey, midXHeadKey, midYHeadKey) = headKey.splitToTuple(8)

      Params(
        encoder = DocumentEncoder.Params.xavierUniformDepthScaled(numLayers, numHeads, embeddingExtent, embeddingMixedExtent, encoderKey),
        decoder = SequenceDecoder.Params.xavierUniformDepthScaled(numLayers, numHeads, embeddingExtent, embeddingExtent, embeddingMixedExtent, decoderKey),
        embedder = NodeEmbedder.Params(
          nodeClass = VocabularyEmbedder.Params.init(nodeClassExtent, partExtent, nodeClassKey),
          construction = VocabularyEmbedder.Params.init(constructionExtent, partExtent, constructionKey),
          startX = VocabularyEmbedder.Params.init(pixelExtent, partExtent, startXKey),
          startY = VocabularyEmbedder.Params.init(pixelExtent, partExtent, startYKey),
          endX = VocabularyEmbedder.Params.init(pixelExtent, partExtent, endXKey),
          endY = VocabularyEmbedder.Params.init(pixelExtent, partExtent, endYKey),
          midX = VocabularyEmbedder.Params.init(pixelExtent, partExtent, midXKey),
          midY = VocabularyEmbedder.Params.init(pixelExtent, partExtent, midYKey),
          projection = AffineLayer.Params.init(nodePartExtent, embeddingExtent, nodeProjectionKey)
        ),
        head = NodeHead.Params(
          nodeClass = AffineLayer.Params.init(embeddingExtent, nodeClassExtent, classHeadKey),
          construction = AffineLayer.Params.init(embeddingExtent, constructionExtent, constructionHeadKey),
          startX = AffineLayer.Params.init(embeddingExtent, pixelExtent, startXHeadKey),
          startY = AffineLayer.Params.init(embeddingExtent, pixelExtent, startYHeadKey),
          endX = AffineLayer.Params.init(embeddingExtent, pixelExtent, endXHeadKey),
          endY = AffineLayer.Params.init(embeddingExtent, pixelExtent, endYHeadKey),
          midX = AffineLayer.Params.init(embeddingExtent, pixelExtent, midXHeadKey),
          midY = AffineLayer.Params.init(embeddingExtent, pixelExtent, midYHeadKey)
        ),
        start = Init.xavierUniform(Axis[Node] -> 1, embeddingExtent, startKey).squeeze(Axis[Node]),
        positions = LearnedAbsolutePositionalInjector.Params.lecunNormal(Axis[Node] -> nodes, embeddingExtent, positionKey)
      )
