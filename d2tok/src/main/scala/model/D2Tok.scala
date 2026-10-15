package d2tok.model

import d2tok.*
import deepwit.attention.KVCache
import deepwit.base.AffineLayer
import deepwit.embedder.LearnedAbsolutePositionalInjector
import deepwit.embedder.VocabularyEmbedder
import documentEncoder.DocumentEncoder
import dimwit.*

import scala.language.implicitConversions

/** Document-to-words model: a language model of records, conditioned on the document.
  *
  * 1. The document (or: image) is embedded by a vision transformer to a sequence of patch embeddings.
  * 2. The words a record is written in are predicted one after the other, each from the document
  *    (1) and the words before it.
  */
class D2Tok[V: IsFloating](params: D2Tok.Params[V]):

  val encodeDocument = DocumentEncoder(params.encoder)

  private val embedWord = VocabularyEmbedder(params.words)
  private val wordPosition = LearnedAbsolutePositionalInjector(params.positions)
  private val decoder = WordDecoder(params.decoder)
  private val nextWord = AffineLayer(params.head)

  /** The scores of the word at every position, read from the words `read` up to it. */
  def logits(document: Tensor3[Width, Height, Channel, V], read: Tensor1[Token, Int32]): Tensor2[Token, Word, V] =
    decoder(encodeDocument(document), wordPosition(read.vmap(Axis[Token])(embedWord)))
      .vmap(Axis[Token])(nextWord)

  // Writing the words down one at a time, as [[logits]] would have them at every position, but
  // projecting at each step only what the step adds.

  /** The encoded document as every step of the decoder reads it. */
  def read(encodedDocument: Tensor2[Patch, Embedding, V]): List[KVCache[Patch, V]] =
    decoder.read(encodedDocument)

  /** The decoder before any word is written. */
  def nothingWritten(tokens: AxisExtent[Token]): List[KVCache[Token, V]] =
    decoder.nothingWritten(tokens)

  /** The scores of the word at `position`, reading `previous` — the start word at the first
    * position, the word before it after that — and the words `written` before it; and `written`
    * with the position among them.
    */
  def answerAt(documentCache: List[KVCache[Patch, V]], written: List[KVCache[Token, V]], position: Tensor0[Int32], previous: Tensor0[Int32]): (Tensor1[Word, V], List[KVCache[Token, V]]) =
    val (answered, nowWritten) = decoder.decode(documentCache, written, position, wordPosition.injectAt(position)(embedWord(previous)))
    (nextWord(answered), nowWritten)

object D2Tok:

  case class Params[V](
      encoder: DocumentEncoder.Params[Embedding, V],
      decoder: WordDecoder.Params[Embedding, Embedding, V],
      words: VocabularyEmbedder.Params[Word, Embedding, V],
      positions: LearnedAbsolutePositionalInjector.Params[Token, Embedding, V],
      head: AffineLayer.Params[Embedding, Word, V]
  )

  object Params:

    given tensorTree: TensorTree[Params[Float32]] = TensorTree.derived
    given tree: TreeOf[Params[Float32], Float32] = TreeOf.derived

    /** @param words How many words the vocabulary holds.
      * @param tokens How many words the longest record of the data is written in, its end among them.
      */
    def init(numLayers: Int, numHeads: Int, embedding: Int, words: Int, tokens: Int, key: Key): Params[Float32] =
      val (encoderKey, decoderKey, wordKey, positionKey, headKey) = key.splitToTuple(5)
      val embeddingExtent = Axis[Embedding] -> embedding
      val embeddingMixedExtent = Axis[DocumentEncoder.EmbeddingMixed] -> embedding * 4
      val wordExtent = Axis[Word] -> words
      Params(
        encoder = DocumentEncoder.Params.xavierUniformDepthScaled(numLayers, numHeads, embeddingExtent, embeddingMixedExtent, encoderKey),
        decoder = WordDecoder.Params.xavierUniformDepthScaled(numLayers, numHeads, embeddingExtent, embeddingExtent, embeddingMixedExtent, decoderKey),
        words = VocabularyEmbedder.Params.init(wordExtent, embeddingExtent, wordKey),
        positions = LearnedAbsolutePositionalInjector.Params.lecunNormal(Axis[Token] -> tokens, embeddingExtent, positionKey),
        head = AffineLayer.Params.init(embeddingExtent, wordExtent, headKey)
      )
