import dataset.EdgeClass
import dataset.NodeClass
import dataset.NodeClasses
import dataset.NodeLink
import dataset.NodePoint
import dataset.Record
import dataset.RecordEdges
import dataset.RecordNodes
import deepwit.base.AffineLayer
import deepwit.embedder.ImageToPatchEmbedder
import deepwit.embedder.LearnedAbsolutePositionalInjector
import deepwit.init.Init
import EdgeScorer.EdgeLogits
import NodeScorer.NodeLogits
import dimwit.*

import scala.annotation.tailrec

/** Document-to-graph transcription,
  * [[https://arxiv.org/abs/2507.08458 A document is worth a structured record]] §3.7, on the
  * l-shape drawings — see `README.md` for the divergences from the paper.
  *
  * The document is embedded patch by patch and attended over by the encoder. Its record is written
  * down in two stages: the nodes it draws, and then the relationships between them. Each stage is
  * fed what it has taken so far and answers, at every position, with one it has not taken yet;
  * the relationships read the nodes as well as the document, since a relationship is a pair of
  * nodes. Nothing is detected and nothing is assembled afterwards: the drawing's record,
  * relationships and all, is what the model writes down.
  */
class D2G[V: IsFloating](params: D2G.Params[V]):

  import D2G.EdgeScores
  import D2G.NodeScores
  import D2G.Scores

  private val patches = ImageToPatchEmbedder(params.patchEmbedder)
  private val encoder = DocumentEncoder(params.encoder)
  private val nodeDecoder = NodeDecoder(params.nodes.decoder)
  private val edgeDecoder = EdgeDecoder(params.edges.decoder)
  private val nodePosition = LearnedAbsolutePositionalInjector(params.nodes.positions)
  private val edgePosition = LearnedAbsolutePositionalInjector(params.edges.positions)
  private val nodeScorer = NodeScorer(params.nodes.scorer)
  private val edgeScorer = EdgeScorer(params.edges.scorer)
  private val embedNodes = NodeEmbedder(params.nodes.embedder, nodeScorer.canvas)
  private val embedEdges = EdgeEmbedder(params.edges.embedder)

  /** What the model scores, given the document and the record taken from it so far.
    *
    * The record is the one the drawing actually holds, so this is the teacher-forced door and
    * belongs to training. Transcription reads [[predictRecord]], which is given no record at all.
    */
  def apply(document: Tensor3[Width, Height, Channel, V], taken: Record[Node, Edge]): Scores[V] =
    predict(encode(document), taken)

  /** The document, once. Transcription reads it at every step, so it is worth keeping. */
  def encode(document: Tensor3[Width, Height, Channel, V]): Tensor2[Patch, Embedding, V] =
    encoder(patches(document))

  def predict(document: Tensor2[Patch, Embedding, V], taken: Record[Node, Edge]): Scores[V] =
    val nodes = taken.nodeClass.shape.extent(Axis[Node])
    val edges = taken.edgeClass.shape.extent(Axis[Edge])
    val (carriedNodes, answeredNodes) =
      nodeDecoder.forTraining(document, nodePosition(embedNodes(taken.nodes)), nodePredictions(nodes))
    val (carriedEdges, answeredEdges) =
      edgeDecoder.forTraining(document, carriedNodes, holdsNode(taken.nodes), edgePosition(embedEdges(taken.edges)), edgePredictions(edges))
    Scores(
      // What a prediction slot answers with is a node, or a relationship, so what it became is
      // read back as one.
      nodes = NodeScores(
        remaining = nodeScorer(answeredNodes.relabel(Axis[NodePrediction] -> Axis[Node])),
        taken = nodeScorer(carriedNodes)
      ),
      edges = EdgeScores(
        remaining = edgeScorer(answeredEdges.relabel(Axis[EdgePrediction] -> Axis[Edge])),
        taken = edgeScorer(carriedEdges)
      )
    )

  /** The record the document holds: the nodes it draws, and then the relationships between them.
    *
    * Nothing but the encoded document goes in, and every step reads back only what the model
    * itself has taken, so no target record can reach it here — which is the whole of transcription,
    * and nothing else in this class is part of it.
    */
  def predictRecord(document: Tensor2[Patch, Embedding, V], nodeSlots: AxisExtent[Node], edgeSlots: AxisExtent[Edge]): Record[Node, Edge] =

    def predictRemainingNodes: RecordNodes[Node] =
      def noNodes(slots: AxisExtent[Node]) =
        val points = Axis[NodePoint] -> NodeClass.maxPoints
        RecordNodes(
          nodeClass = Tensor1(slots, VType[Int32]).fill(NodeClass.NoNode.id),
          xs = Tensor2(slots, points, VType[Float32]).fill(0f),
          ys = Tensor2(slots, points, VType[Float32]).fill(0f)
        )

      def joined(taken: RecordNodes[Node], next: RecordNodes[Node]) =
        RecordNodes(
          nodeClass = concatenate(taken.nodeClass, next.nodeClass, Axis[Node]),
          xs = concatenate(taken.xs, next.xs, Axis[Node]),
          ys = concatenate(taken.ys, next.ys, Axis[Node])
        )

      /** The node the model answers slot `at` with, having taken the nodes before it. */
      def nextRemainingNode(taken: RecordNodes[Node], at: Int) =
        val context = concatenate(embedNodes(taken), params.nodes.token.prependAxis(Axis[Node]), Axis[Node])
        val placed = nodePosition.injectToPrefix(context)
        val (_, answered) = nodeDecoder.forTranscription(document, placed.slice(Axis[Node].at(0 until at)), placed.slice(Axis[Node].at(at)))
        nodeScorer.decide(nodeScorer(answered.prependAxis(Axis[Node])))

      @tailrec def untilNodesEnd(taken: RecordNodes[Node]): RecordNodes[Node] =
        val at = taken.nodeClass.shape(Axis[Node])
        if at == nodeSlots.size then taken
        else
          val next = nextRemainingNode(taken, at)
          if next.nodeClass.toArray.head == NodeClass.NoNode.id then taken
          else untilNodesEnd(joined(taken, next))

      untilNodesEnd(noNodes(Axis[Node] -> 0))

    /** Every relationship between those nodes, predicted the same way. The nodes are read as the
      * node decoder carries them, since that is what an [[EdgeDecoder]] relates; the prediction
      * that door answers with is thrown away, nothing being predicted there.
      */
    def predictRemainingEdges(nodes: RecordNodes[Node]): RecordEdges[Edge] =
      val related = nodeDecoder.forTranscription(document, nodePosition.injectToPrefix(embedNodes(nodes)), params.nodes.token)._1
      val holds = holdsNode(nodes)

      def noEdges(slots: AxisExtent[Edge]) =
        RecordEdges(
          edgeClass = Tensor1(slots, VType[Int32]).fill(EdgeClass.NoEdge.id),
          links = Tensor2(slots, Axis[NodeLink] -> EdgeClass.maxLinks, VType[Int32]).fill(0)
        )

      def joined(taken: RecordEdges[Edge], next: RecordEdges[Edge]) =
        RecordEdges(
          edgeClass = concatenate(taken.edgeClass, next.edgeClass, Axis[Edge]),
          links = concatenate(taken.links, next.links, Axis[Edge])
        )

      /** The relationship the model answers slot `at` with, having taken the ones before it. */
      def nextRemainingEdge(taken: RecordEdges[Edge], at: Int) =
        val context = concatenate(embedEdges(taken), params.edges.token.prependAxis(Axis[Edge]), Axis[Edge])
        val placed = edgePosition.injectToPrefix(context)
        val answered = edgeDecoder.forTranscription(document, related, holds, placed.slice(Axis[Edge].at(0 until at)), placed.slice(Axis[Edge].at(at)))
        edgeScorer.decide(edgeScorer(answered.prependAxis(Axis[Edge])))

      @tailrec def untilEdgesEnd(taken: RecordEdges[Edge]): RecordEdges[Edge] =
        val at = taken.edgeClass.shape(Axis[Edge])
        if at == edgeSlots.size then taken
        else
          val next = nextRemainingEdge(taken, at)
          if next.edgeClass.toArray.head == EdgeClass.NoEdge.id then taken
          else untilEdgesEnd(joined(taken, next))

      untilEdgesEnd(noEdges(Axis[Edge] -> 0))

    val nodes = predictRemainingNodes
    Record(nodes, predictRemainingEdges(nodes))

  /** Which node slots hold a node: the positions past the record are there to make every record
    * the same shape, and relate nothing. Both doors ask, since both hand nodes to an
    * [[EdgeDecoder]].
    */
  private def holdsNode(nodes: RecordNodes[Node]): Tensor1[Node, Bool] =
    val drawn = NodeClass.indicator(VType[Float32])(_.isDrawn).take(Axis[NodeClasses])(nodes.nodeClass)
    drawn > Tensor.like(drawn).fill(0f)

  /** The same `<P>` token at every node slot, told apart only by the positional encoding of the
    * node it answers for and by what it may attend to.
    */
  private def nodePredictions(nodes: AxisExtent[Node]): Tensor2[NodePrediction, Embedding, V] =
    val tokens = params.nodes.token.broadcastTo(Shape2(nodes, params.nodes.token.shape.extent(Axis[Embedding])))
    nodePosition(tokens).relabel(Axis[Node] -> Axis[NodePrediction])

  /** The same, at every relationship slot. */
  private def edgePredictions(edges: AxisExtent[Edge]): Tensor2[EdgePrediction, Embedding, V] =
    val tokens = params.edges.token.broadcastTo(Shape2(edges, params.edges.token.shape.extent(Axis[Embedding])))
    edgePosition(tokens).relabel(Axis[Edge] -> Axis[EdgePrediction])

object D2G:

  /** What the model scores per node slot: the remaining node its prediction embedding answers
    * with, and the taken node its node embedding passed through.
    */
  case class NodeScores[V](remaining: NodeLogits[V], taken: NodeLogits[V])

  /** The same per relationship slot. */
  case class EdgeScores[V](remaining: EdgeLogits[V], taken: EdgeLogits[V])

  /** What the model scores for a document: both halves of the record it would write down. */
  case class Scores[V](nodes: NodeScores[V], edges: EdgeScores[V])

  /** Everything the nodes of a record are written down with: what reads them, what embeds them,
    * what scores them, the `<P>` token every prediction embedding starts as, and where each slot
    * sits.
    */
  case class NodeParams[V](
      decoder: NodeDecoder.Params[Embedding, Embedding, V],
      embedder: NodeEmbedder.Params[V],
      scorer: NodeScorer.Params[V],
      token: Tensor1[Embedding, V],
      positions: LearnedAbsolutePositionalInjector.Params[Node, Embedding, V]
  )

  object NodeParams:

    given tensorTree: TensorTree[NodeParams[Float32]] = TensorTree.derived
    given tree: TreeOf[NodeParams[Float32], Float32] = TreeOf.derived

  /** The same for the relationships between them. */
  case class EdgeParams[V](
      decoder: EdgeDecoder.Params[Embedding, Embedding, V],
      embedder: EdgeEmbedder.Params[V],
      scorer: EdgeScorer.Params[V],
      token: Tensor1[Embedding, V],
      positions: LearnedAbsolutePositionalInjector.Params[Edge, Embedding, V]
  )

  object EdgeParams:

    given tensorTree: TensorTree[EdgeParams[Float32]] = TensorTree.derived
    given tree: TreeOf[EdgeParams[Float32], Float32] = TreeOf.derived

  case class Params[V](
      patchEmbedder: ImageToPatchEmbedder.Params[Width, Height, Channel, Embedding, V],
      encoder: DocumentEncoder.Params[Embedding, V],
      nodes: NodeParams[V],
      edges: EdgeParams[V]
  )

  object Params:

    given tensorTree: TensorTree[Params[Float32]] = TensorTree.derived
    given tree: TreeOf[Params[Float32], Float32] = TreeOf.derived

    /** @param nodes  How many nodes of a record the model can hold. One more than the most any
      *               record of the data draws, so that the last prediction embedding has somewhere
      *               to say the nodes have ended.
      * @param edges  The same for the relationships between them.
      * @param canvas The width of the drawing, which is how many pixels a coordinate chooses from.
      */
    def init(
        numLayers: Int,
        numHeads: Int,
        embedding: Int,
        nodes: Int,
        edges: Int,
        patchSize: Int,
        canvas: Int,
        key: Key
    ): Params[Float32] =
      val (patchKey, encoderKey, decoderKey, embedderKey, scorerKey, tokenKey, positionKey) = key.splitToTuple(7)

      val embeddingExtent = Axis[Embedding] -> embedding
      val embeddingMixedExtent = Axis[EmbeddingMixed] -> embedding * 4
      val partExtent = Axis[PartEmbedding] -> embedding / 8
      val nodeClassExtent = Axis[dataset.NodeClasses] -> NodeClass.values.length
      val edgeClassExtent = Axis[dataset.EdgeClasses] -> EdgeClass.values.length
      val pixelExtent = Axis[Pixel] -> canvas
      val nodeExtent = Axis[Node] -> nodes
      val edgeExtent = Axis[Edge] -> edges
      val linkedExtent = Axis[LinkedNode] -> nodes
      val pointExtent = Axis[NodePoint] -> NodeClass.maxPoints
      val linkExtent = Axis[NodeLink] -> EdgeClass.maxLinks
      val nodePartExtent = Axis[NodePart |*| PartEmbedding] -> (1 + 2 * NodeClass.maxPoints) * partExtent.size
      val edgePartExtent = Axis[EdgePart |*| PartEmbedding] -> (1 + EdgeClass.maxLinks) * partExtent.size

      val (nodeDecoderKey, edgeDecoderKey) = decoderKey.splitToTuple(2)
      val (nodeEmbedderKey, edgeEmbedderKey) = embedderKey.splitToTuple(2)
      val (xKey, yKey, nodeClassKey, nodeProjectionKey) = nodeEmbedderKey.splitToTuple(4)
      val (linkKey, edgeClassKey, edgeProjectionKey) = edgeEmbedderKey.splitToTuple(3)
      val (nodeHeadKey, edgeHeadKey) = scorerKey.splitToTuple(2)
      val (classHeadKey, xHeadKey, yHeadKey) = nodeHeadKey.splitToTuple(3)
      val (edgeClassHeadKey, linkHeadKey) = edgeHeadKey.splitToTuple(2)
      val (nodeTokenKey, edgeTokenKey) = tokenKey.splitToTuple(2)
      val (nodePositionKey, edgePositionKey) = positionKey.splitToTuple(2)

      def perCarried[Carries: Label, In, Out](carries: AxisExtent[Carries], key: Key)(
          weights: Key => Tensor2[In, Out, Float32]
      )(using Label[In], Label[Out]): Tensor3[Carries, In, Out, Float32] =
        stack(key.split(carries.size).map(weights).toSeq, carries.axis)

      def head[Carries: Label, Values: Label](carries: AxisExtent[Carries], values: AxisExtent[Values], key: Key) =
        ScorerHead(
          projection = perCarried(carries, key)(Init.xavierUniform(embeddingExtent, values, _)),
          bias = Tensor2(carries, values, VType[Float32]).fill(0f)
        )

      Params(
        patchEmbedder = ImageToPatchEmbedder.Params.init(
          Axis[Width] -> patchSize,
          Axis[Height] -> patchSize,
          Axis[Channel] -> 1,
          embeddingExtent,
          patchKey
        ),
        encoder = DocumentEncoder.Params.xavierUniformDepthScaled(numLayers, numHeads, embeddingExtent, embeddingMixedExtent, encoderKey),
        nodes = NodeParams(
          decoder = NodeDecoder.Params.xavierUniformDepthScaled(numLayers, numHeads, embeddingExtent, embeddingExtent, embeddingMixedExtent, nodeDecoderKey),
          embedder = NodeEmbedder.Params(
            nodeClass = Init.xavierUniform(nodeClassExtent, partExtent, nodeClassKey),
            xs = perCarried(pointExtent, xKey)(Init.xavierUniform(pixelExtent, partExtent, _)),
            ys = perCarried(pointExtent, yKey)(Init.xavierUniform(pixelExtent, partExtent, _)),
            projection = AffineLayer.Params.init(nodePartExtent, embeddingExtent, nodeProjectionKey)
          ),
          scorer = NodeScorer.Params(
            nodeClass = AffineLayer.Params.init(embeddingExtent, nodeClassExtent, classHeadKey),
            xs = head(pointExtent, pixelExtent, xHeadKey),
            ys = head(pointExtent, pixelExtent, yHeadKey)
          ),
          token = Init.xavierUniformVector(embeddingExtent, nodeTokenKey),
          positions = LearnedAbsolutePositionalInjector.Params.lecunNormal(nodeExtent, embeddingExtent, nodePositionKey)
        ),
        edges = EdgeParams(
          decoder = EdgeDecoder.Params.xavierUniformDepthScaled(numLayers, numHeads, embeddingExtent, embeddingExtent, embeddingMixedExtent, edgeDecoderKey),
          embedder = EdgeEmbedder.Params(
            edgeClass = Init.xavierUniform(edgeClassExtent, partExtent, edgeClassKey),
            links = perCarried(linkExtent, linkKey)(Init.xavierUniform(linkedExtent, partExtent, _)),
            projection = AffineLayer.Params.init(edgePartExtent, embeddingExtent, edgeProjectionKey)
          ),
          scorer = EdgeScorer.Params(
            edgeClass = AffineLayer.Params.init(embeddingExtent, edgeClassExtent, edgeClassHeadKey),
            links = head(linkExtent, linkedExtent, linkHeadKey)
          ),
          token = Init.xavierUniformVector(embeddingExtent, edgeTokenKey),
          positions = LearnedAbsolutePositionalInjector.Params.lecunNormal(edgeExtent, embeddingExtent, edgePositionKey)
        )
      )
