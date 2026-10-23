package d2tok.floorplangraph.model

import d2tok.*
import d2tok.floorplangraph.*
import d2tok.model.D2Tok
import deepwit.attention.KVCache
import deepwit.base.AffineLayer
import deepwit.embedder.LearnedAbsolutePositionalInjector
import deepwit.embedder.VocabularyEmbedder
import documentEncoder.DocumentEncoder
import dimwit.*

import scala.language.implicitConversions

/** Document-to-graph model: a [[D2Tok]] writes a floor plan's nodes in words, and a second decoder
  * writes the relationships between them in words of their own.
  *
  * 1. The document (or: image) is embedded by a vision transformer to a sequence of patch embeddings.
  * 2. The record is written in two stages:
  *   a. the words of the nodes, each from the document (1) and the words before it, by the [[D2Tok]];
  *   b. the words of the relationships, each from the document (1), the nodes as the [[D2Tok]]
  *      carried them (2a), and the relationship words before it.
  */
class D2TokGraph[V: IsFloating](params: D2TokGraph.Params[V]):

  /** The model of the nodes, whose encoding of the document the relationships read too. */
  val nodes = D2Tok(params.nodes)
  val encodeDocument = nodes.encodeDocument

  private val numberNode = LearnedAbsolutePositionalInjector(params.nodeNumbers)
  private val embedWord = VocabularyEmbedder(params.words)
  private val wordPosition = LearnedAbsolutePositionalInjector(params.positions)
  private val decoder = RelationshipDecoder(params.decoder)
  private val nextWord = AffineLayer(params.head)

  /** The scores of the node words and of the relationship words at every position, each read from
    * the words of its kind up to it — `readNodes` and `readRelationships` — and the relationships
    * from the nodes `present` in `readNodes` besides.
    *
    * @param perNode How many words every node is written in.
    */
  def logits(
      document: Tensor3[Width, Height, Channel, V],
      readNodes: Tensor1[Token, Int32],
      perNode: Int,
      present: Tensor1[Node, Bool],
      readRelationships: Tensor1[RelationshipToken, Int32]
  ): (Tensor2[Token, Word, V], Tensor2[RelationshipToken, Word, V]) =
    val encoded = encodeDocument(document)
    val carried = nodes.carried(encoded, readNodes)
    val relationships = decoder(encoded, numbered(carried, perNode, present), present, wordPosition(readRelationships.vmap(Axis[RelationshipToken])(embedWord)))
    (nodes.scored(carried), relationships.vmap(Axis[RelationshipToken])(nextWord))

  /** The nodes as the relationships read them: each as the node decoder carried the word read after
    * it, which has seen every word of the node, and numbered as the relationship words name it.
    */
  def numbered(carried: Tensor2[Token, Embedding, V], perNode: Int, present: Tensor1[Node, Bool]): Tensor2[Node, Embedding, V] =
    val slots = present.shape.extent(Axis[Node])
    val afterEachNode = carried
      .slice(Axis[Token].at(1 to slots.size * perNode))
      .relabel(Axis[Token] -> Axis[Node |*| Part])
      .unflatten(Axis[Node |*| Part], Shape(slots, Axis[Part] -> perNode))
      .slice(Axis[Part].at(perNode - 1))
    numberNode(afterEachNode)

  // Writing the relationship words down one at a time, as [[logits]] would have them at every
  // position, but projecting at each step only what the step adds.

  /** The encoded document and the nodes as every step of the relationship decoder reads them. */
  def read(encodedDocument: Tensor2[Patch, Embedding, V], nodes: Tensor2[Node, Embedding, V], present: Tensor1[Node, Bool]): List[RelationshipSources[V]] =
    decoder.read(encodedDocument, nodes, present)

  /** The relationship decoder before any word is written. */
  def nothingWritten(tokens: AxisExtent[RelationshipToken]): List[KVCache[RelationshipToken, V]] =
    decoder.nothingWritten(tokens)

  /** The scores of the relationship word at `position`, reading `previous` — the start word at the
    * first position, the word before it after that — and the words `written` before it; and
    * `written` with the position among them.
    */
  def answerAt(sources: List[RelationshipSources[V]], written: List[KVCache[RelationshipToken, V]], position: Tensor0[Int32], previous: Tensor0[Int32]): (Tensor1[Word, V], List[KVCache[RelationshipToken, V]]) =
    val (answered, nowWritten) = decoder.decode(sources, written, position, wordPosition.injectAt(position)(embedWord(previous)))
    (nextWord(answered), nowWritten)

object D2TokGraph:

  case class Params[V](
      nodes: D2Tok.Params[V],
      nodeNumbers: LearnedAbsolutePositionalInjector.Params[Node, Embedding, V],
      words: VocabularyEmbedder.Params[Word, Embedding, V],
      positions: LearnedAbsolutePositionalInjector.Params[RelationshipToken, Embedding, V],
      decoder: RelationshipDecoder.Params[V],
      head: AffineLayer.Params[Embedding, Word, V]
  )

  object Params:

    given tensorTree: TensorTree[Params[Float32]] = TensorTree.derived
    given tree: TreeOf[Params[Float32], Float32] = TreeOf.derived

    /** @param nodeWords How many words the vocabulary of the nodes holds.
      * @param nodeTokens How many words the nodes of the longest record are written in, their end among them.
      * @param nodes How many nodes the longest record holds.
      * @param relationshipWords How many words the vocabulary of the relationships holds.
      * @param relationshipTokens How many words the relationships of the longest record are written in, their end among them.
      */
    def init(numLayers: Int, numHeads: Int, embedding: Int, nodeWords: Int, nodeTokens: Int, nodes: Int, relationshipWords: Int, relationshipTokens: Int, key: Key): Params[Float32] =
      val (nodesKey, numberKey, wordKey, positionKey, decoderKey, headKey) = key.splitToTuple(6)
      val embeddingExtent = Axis[Embedding] -> embedding
      val wordExtent = Axis[Word] -> relationshipWords
      Params(
        nodes = D2Tok.Params.init(numLayers, numHeads, embedding, nodeWords, nodeTokens, nodesKey),
        nodeNumbers = LearnedAbsolutePositionalInjector.Params.lecunNormal(Axis[Node] -> nodes, embeddingExtent, numberKey),
        words = VocabularyEmbedder.Params.init(wordExtent, embeddingExtent, wordKey),
        positions = LearnedAbsolutePositionalInjector.Params.lecunNormal(Axis[RelationshipToken] -> relationshipTokens, embeddingExtent, positionKey),
        decoder = RelationshipDecoder.Params.xavierUniformDepthScaled(numLayers, numHeads, embeddingExtent, Axis[DocumentEncoder.EmbeddingMixed] -> embedding * 4, decoderKey),
        head = AffineLayer.Params.init(embeddingExtent, wordExtent, headKey)
      )
