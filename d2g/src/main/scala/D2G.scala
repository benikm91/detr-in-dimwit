import dataset.NodeClass
import dataset.NodeLink
import dataset.NodePoint
import dataset.Record
import deepwit.base.AffineLayer
import deepwit.embedder.ImageToPatchEmbedder
import deepwit.embedder.LearnedAbsolutePositionalInjector
import dimwit.*

import deepwit.init.Init

/** Document-to-graph transcription,
  * [[https://arxiv.org/abs/2507.08458 A document is worth a structured record]] §3.7, on the
  * l-shape drawings — see `README.md` for the divergences from the paper.
  *
  * The document is embedded patch by patch and attended over by the encoder. The decoder is fed
  * the record's nodes in a random order and answers, at every position, with one of the nodes it
  * has not taken yet. Nothing is detected and nothing is assembled afterwards: the drawing's
  * record, relationships and all, is what the model writes down.
  */
class D2G[V: IsFloating](params: D2G.Params[V]):

  import D2G.Patch
  import D2G.Prediction

  private val patches = ImageToPatchEmbedder(params.patchEmbedder)
  private val encoder = DocumentEncoder(Axis[Patch], params.encoder)
  private val decoder = RecordDecoder(Axis[Patch], Axis[DecoderSlot], params.decoder)
  private val position = LearnedAbsolutePositionalInjector(params.positions)
  private val scorer = NodeScorer(params.scorer)
  private val embed = NodeEmbedder(params.embedder, scorer.canvas)

  /** What the model scores, given the document and the nodes taken from its record so far. */
  def apply(document: Tensor3[Width, Height, Channel, V], taken: Record[Node]): Prediction[V] =
    predict(encode(document), taken)

  /** The document, once. Transcription reads it at every step, so it is worth keeping. */
  def encode(document: Tensor3[Width, Height, Channel, V]): Tensor2[Patch, Embedding, V] =
    encoder(patches(document))

  def predict(document: Tensor2[Patch, Embedding, V], taken: Record[Node]): Prediction[V] =
    val nodes = taken.nodeClass.shape.extent(Axis[Node])
    val sequence = DecoderSequence.join(position(embed(taken)), predictionEmbeddings(nodes))
    val decoded = decoder(document, sequence, DecoderSequence.mask(taken))
    Prediction(
      remaining = scorer(DecoderSequence.remaining(decoded)),
      taken = scorer(DecoderSequence.taken(decoded))
    )

  /** One step of transcription: what every position would take next, of which the loop that
    * drives it keeps only the one it is at.
    */
  def step(document: Tensor2[Patch, Embedding, V], taken: Record[Node]): Record[Node] =
    scorer.decide(predict(document, taken).remaining)

  /** The same `<P>` token everywhere, told apart only by the positional encoding of the node it
    * answers for and by what it may attend to.
    */
  private def predictionEmbeddings(nodes: AxisExtent[Node]): Tensor2[Node, Embedding, V] =
    position(params.predictionToken.broadcastTo(Shape2(nodes, params.predictionToken.shape.extent(Axis[Embedding]))))

object D2G:

  /** The document flattened into the encoder's sequence of patches. */
  type Patch = Width |*| Height

  /** Per position: the remaining node its prediction embedding answers with, and the taken node
    * its node embedding passed through.
    */
  case class Prediction[V](remaining: NodeLogits[V], taken: NodeLogits[V])

  case class Params[V](
      patchEmbedder: ImageToPatchEmbedder.Params[Width, Height, Channel, Embedding, V],
      encoder: DocumentEncoder.Params[Embedding, V],
      decoder: RecordDecoder.Params[Embedding, Embedding, V],
      embedder: NodeEmbedder.Params[V],
      scorer: NodeScorer.Params[V],
      predictionToken: Tensor1[Embedding, V],
      positions: LearnedAbsolutePositionalInjector.Params[Node, Embedding, V]
  )

  object Params:

    given tensorTree: TensorTree[Params[Float32]] = TensorTree.derived
    given tree: TreeOf[Params[Float32], Float32] = TreeOf.derived

    /** @param nodes  How many nodes of a record the model can hold. One more than the longest
      *               record of the data, so that the last prediction embedding has somewhere to
      *               say the record has ended.
      * @param canvas The width of the drawing, which is how many pixels a coordinate chooses from.
      */
    def init(
        numLayers: Int,
        numHeads: Int,
        embedding: Int,
        nodes: Int,
        patchSize: Int,
        canvas: Int,
        key: Key
    ): Params[Float32] =
      val (patchKey, encoderKey, decoderKey, embedderKey, scorerKey, tokenKey, positionKey) = key.splitToTuple(7)

      val embeddingExtent = Axis[Embedding] -> embedding
      val embeddingMixedExtent = Axis[EmbeddingMixed] -> embedding * 4
      val partExtent = Axis[PartEmbedding] -> embedding / 8
      val classExtent = Axis[dataset.NodeClasses] -> NodeClass.values.length
      val pixelExtent = Axis[Pixel] -> canvas
      val nodeExtent = Axis[Node] -> nodes
      val linkedExtent = Axis[LinkedNode] -> nodes
      val pointExtent = Axis[NodePoint] -> NodeClass.maxPoints
      val linkExtent = Axis[NodeLink] -> NodeClass.maxLinks
      val parts = 1 + 2 * NodeClass.maxPoints + NodeClass.maxLinks
      val nodePartExtent = Axis[NodePart |*| PartEmbedding] -> parts * partExtent.size

      val (classKey, xKey, yKey, linkKey, projectionKey) = embedderKey.splitToTuple(5)
      val (classHeadKey, xHeadKey, yHeadKey, linkHeadKey) = scorerKey.splitToTuple(4)

      def perCarried[Carries: Label, In, Out](carries: AxisExtent[Carries], key: Key)(
          weights: Key => Tensor2[In, Out, Float32]
      )(using Label[In], Label[Out]): Tensor3[Carries, In, Out, Float32] =
        stack(key.split(carries.size).map(weights).toSeq, carries.axis)

      def head[Carries: Label, Values: Label](carries: AxisExtent[Carries], values: AxisExtent[Values], key: Key) =
        NodeScorer.Head(
          projection = perCarried(carries, key)(Init.xavierUniform(embeddingExtent, values, _)),
          bias = Tensor2(carries, values, VType[Float32]).fill(0f)
        )

      Params(
        patchEmbedder = ImageToPatchEmbedder.Params.xavierUniform(
          Axis[Width] -> patchSize,
          Axis[Height] -> patchSize,
          Axis[Channel] -> 1,
          embeddingExtent,
          patchKey
        ),
        encoder = DocumentEncoder.Params.xavierUniformDepthScaled(numLayers, numHeads, embeddingExtent, embeddingMixedExtent, encoderKey),
        decoder = RecordDecoder.Params.xavierUniformDepthScaled(numLayers, numHeads, embeddingExtent, embeddingExtent, embeddingMixedExtent, decoderKey),
        embedder = NodeEmbedder.Params(
          nodeClass = Init.xavierUniform(classExtent, partExtent, classKey),
          xs = perCarried(pointExtent, xKey)(Init.xavierUniform(pixelExtent, partExtent, _)),
          ys = perCarried(pointExtent, yKey)(Init.xavierUniform(pixelExtent, partExtent, _)),
          links = perCarried(linkExtent, linkKey)(Init.xavierUniform(linkedExtent, partExtent, _)),
          projection = AffineLayer.Params.xavierUniform(nodePartExtent, embeddingExtent, projectionKey)
        ),
        scorer = NodeScorer.Params(
          nodeClass = AffineLayer.Params.xavierUniform(embeddingExtent, classExtent, classHeadKey),
          xs = head(pointExtent, pixelExtent, xHeadKey),
          ys = head(pointExtent, pixelExtent, yHeadKey),
          links = head(linkExtent, linkedExtent, linkHeadKey)
        ),
        predictionToken = Init.xavierUniformVector(embeddingExtent, tokenKey),
        positions = LearnedAbsolutePositionalInjector.Params.lecunNormal(nodeExtent, embeddingExtent, positionKey)
      )
