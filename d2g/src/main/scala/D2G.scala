import dataset.NodeClass
import dataset.NodeLink
import dataset.NodePoint
import dataset.Record
import deepwit.base.AffineLayer
import deepwit.embedder.ImageToPatchEmbedder
import deepwit.embedder.LearnedAbsolutePositionalInjector
import dimwit.*

import scala.annotation.tailrec

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

  import D2G.Scores

  private val patches = ImageToPatchEmbedder(params.patchEmbedder)
  private val encoder = DocumentEncoder(params.encoder)
  private val decoder = RecordDecoder(params.decoder)
  private val position = LearnedAbsolutePositionalInjector(params.positions)
  private val scorer = NodeScorer(params.scorer)
  private val embed = NodeEmbedder(params.embedder, scorer.canvas)

  /** What the model scores, given the document and the nodes taken from its record so far.
    *
    * The nodes are the ones the record actually holds, so this is the teacher-forced door and
    * belongs to training. Transcription reads [[transcribe]], which is given no record at all.
    */
  def apply(document: Tensor3[Width, Height, Channel, V], taken: Record[Node]): Scores[V] =
    predict(encode(document), taken)

  /** The document, once. Transcription reads it at every step, so it is worth keeping. */
  def encode(document: Tensor3[Width, Height, Channel, V]): Tensor2[Patch, Embedding, V] =
    encoder(patches(document))

  def predict(document: Tensor2[Patch, Embedding, V], taken: Record[Node]): Scores[V] =
    val nodes = taken.nodeClass.shape.extent(Axis[Node])
    val (carried, answered) = decoder.forTraining(document, position(embed(taken)), predictionEmbeddings(nodes))
    // What a prediction slot answers with is a node, so what it became is read back as one.
    Scores(
      remainingPredictions = scorer(answered.relabel(Axis[RecordDecoder.Prediction] -> Axis[Node])),
      takenNodes = scorer(carried)
    )

  /** The record the document holds: every remaining node predicted after the ones taken before it,
    * up to `slots` of them or to the node the model ends the record with.
    *
    * Nothing but the encoded document goes in, and every step reads back only what the model
    * itself has taken, so no target record can reach it here.
    */
  def predictRemainingNodes(document: Tensor2[Patch, Embedding, V], slots: AxisExtent[Node]): Record[Node] =
    @tailrec def untilRecordEnds(taken: Record[Node]): Record[Node] =
      val at = taken.nodeClass.shape(Axis[Node])
      if at == slots.size then taken
      else
        val next = nextRemainingNode(document, taken, at)
        if next.nodeClass.toArray.head == NodeClass.NoNode.id then taken
        else untilRecordEnds(joined(taken, next))
    untilRecordEnds(noNodes)

  /** The node the model answers slot `at` with, having taken the nodes before it. */
  private def nextRemainingNode(document: Tensor2[Patch, Embedding, V], taken: Record[Node], at: Int): Record[Node] =
    val context = concatenate(embed(taken), params.predictionToken.prependAxis(Axis[Node]), Axis[Node])
    val placed = position.injectToPrefix(context)
    val answered = decoder.forTranscription(document, placed.slice(Axis[Node].at(0 until at)), placed.slice(Axis[Node].at(at)))
    scorer.decide(scorer(answered.prependAxis(Axis[Node])))

  /** A record with nothing taken in it yet, which is where transcription starts. */
  private def noNodes: Record[Node] =
    val none = Axis[Node] -> 0
    Record(
      nodeClass = Tensor1(none, VType[Int32]).fill(0),
      xs = Tensor2(none, Axis[NodePoint] -> NodeClass.maxPoints, VType[Float32]).fill(0f),
      ys = Tensor2(none, Axis[NodePoint] -> NodeClass.maxPoints, VType[Float32]).fill(0f),
      links = Tensor2(none, Axis[NodeLink] -> NodeClass.maxLinks, VType[Int32]).fill(0)
    )

  /** The record with one more node taken into the slot after its last. */
  private def joined(taken: Record[Node], next: Record[Node]): Record[Node] =
    Record(
      nodeClass = concatenate(taken.nodeClass, next.nodeClass, Axis[Node]),
      xs = concatenate(taken.xs, next.xs, Axis[Node]),
      ys = concatenate(taken.ys, next.ys, Axis[Node]),
      links = concatenate(taken.links, next.links, Axis[Node])
    )

  /** The same `<P>` token everywhere, told apart only by the positional encoding of the node it
    * answers for and by what it may attend to.
    */
  private def predictionEmbeddings(nodes: AxisExtent[Node]): Tensor2[RecordDecoder.Prediction, Embedding, V] =
    val tokens = params.predictionToken.broadcastTo(Shape2(nodes, params.predictionToken.shape.extent(Axis[Embedding])))
    position(tokens).relabel(Axis[Node] -> Axis[RecordDecoder.Prediction])

object D2G:

  /** What the model scores per slot: the remaining node its prediction embedding answers with,
    * and the taken node its node embedding passed through — the answer and, beside it, what the
    * auxiliary term of [[RemainingNodeLoss]] holds the carrying half to.
    */
  case class Scores[V](remainingPredictions: NodeLogits[V], takenNodes: NodeLogits[V])

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
          projection = AffineLayer.Params.init(nodePartExtent, embeddingExtent, projectionKey)
        ),
        scorer = NodeScorer.Params(
          nodeClass = AffineLayer.Params.init(embeddingExtent, classExtent, classHeadKey),
          xs = head(pointExtent, pixelExtent, xHeadKey),
          ys = head(pointExtent, pixelExtent, yHeadKey),
          links = head(linkExtent, linkedExtent, linkHeadKey)
        ),
        predictionToken = Init.xavierUniformVector(embeddingExtent, tokenKey),
        positions = LearnedAbsolutePositionalInjector.Params.lecunNormal(nodeExtent, embeddingExtent, positionKey)
      )
