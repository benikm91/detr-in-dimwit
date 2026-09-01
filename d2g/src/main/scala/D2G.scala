import dataset.EdgeClass
import dataset.NodeClass
import dataset.NodeClasses
import dataset.Record
import dataset.RecordBatch
import dataset.RecordEdges
import dataset.RecordNodes
import deepwit.base.AffineLayer
import deepwit.embedder.ImageToPatchEmbedder
import deepwit.embedder.LearnedAbsolutePositionalInjector
import deepwit.init.Init
import EdgeScorer.EdgeLogits
import NodeScorer.NodeLogits
import dimwit.*

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
  private val nodeScorer = NodeScorer(params.nodes.scorer)
  private val nodePosition = LearnedAbsolutePositionalInjector(params.nodes.positions)
  private val edgePosition = LearnedAbsolutePositionalInjector(params.edges.positions)
  private val edgeScorer = EdgeScorer(params.edges.scorer)
  private val embedNodes = NodeEmbedder(params.nodes.embedder, nodeScorer.canvas)
  private val embedEdges = EdgeEmbedder(params.edges.embedder)

  /** What the model scores, given the document and the record taken from it so far.
    *
    * The record is the one the drawing actually holds, so this is the teacher-forced door and
    * belongs to training. Transcription reads [[predictRecords]], which is given no record at all.
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

  /** The records a batch of documents hold: the nodes each drawing draws, and then the
    * relationships between them.
    *
    * Nothing but the encoded documents goes in, and every step reads back only what the model
    * itself has taken, so no target record can reach it here — which is the whole of
    * transcription, and nothing else in this class is part of it.
    *
    * The drawings are written down in lockstep: every one of them takes its first node, then its
    * second, and so on for as many slots as a record has. A drawing that has answered
    * [[NodeClass.NoNode]] takes nothing more, and the slots it would have filled hold nothing —
    * which is what a position a record does not reach holds anyway, so the record a drawing ends
    * up with is the one it would have been given had it been transcribed on its own. Its
    * relationships follow the same way.
    *
    * Stepping in lockstep is what makes a step the same piece of work whatever the drawings
    * answer, and so something `jit` can compile once and `vmap` can spread over the batch: where
    * a transcription stops becomes a value rather than a branch, and nothing is read back to the
    * host until the whole batch is written down.
    */
  def predictRecords[Drawing: Label](
      documents: Tensor3[Drawing, Patch, Embedding, V],
      nodeSlots: AxisExtent[Node],
      edgeSlots: AxisExtent[Edge]
  ): RecordBatch[Drawing, Node, Edge] =

    val drawings = documents.shape.extent(Axis[Drawing])

    /** Nothing taken yet: no nodes, no relationships, in as many drawings as there are. */
    val nothingTaken =
      def held[L: Label](slots: AxisExtent[L]) = Shape2(drawings, slots)
      val (noNodes, noEdges) = (held(Axis[Node] -> 0), held(Axis[Edge] -> 0))
      def nowhere = Tensor(noNodes, VType[Float32]).fill(0f)
      def nothing = Tensor(noEdges, VType[Int32]).fill(0)
      RecordBatch[Drawing, Node, Edge](
        nodeClass = Tensor(noNodes, VType[Int32]).fill(NodeClass.NoNode.id),
        startX = nowhere,
        startY = nowhere,
        endX = nowhere,
        endY = nowhere,
        edgeClass = Tensor(noEdges, VType[Int32]).fill(EdgeClass.NoEdge.id),
        subject = nothing,
        obj = nothing
      )

    /** Every drawing still writing down what it is asked for, which at the start is all of them.
      * The nodes and the relationships are two stages, and a drawing writes both.
      */
    val allTaking = Tensor1(drawings, VType[Bool]).fill(true)

    /** One more node slot, taken by every drawing that is still taking nodes. */
    def takeNode(taken: RecordBatch[Drawing, Node, Edge], taking: Tensor1[Drawing, Bool]) =
      val (nodeClass, startX, startY, endX, endY) =
        zipvmap(Axis[Drawing])(documents, taken.nodeClass, taken.startX, taken.startY, taken.endX, taken.endY):
          case (document, nodeClass, startX, startY, endX, endY) =>
            nextNode(document, RecordNodes(nodeClass, startX, startY, endX, endY))
      // A drawing takes the node if it was still taking any and what it answered is a node. The
      // one that says the nodes have ended is not taken either, so a record holds what it drew
      // and stops there.
      val slot = taking.appendAxis(Axis[Node])
      val noNode = Tensor.like(nodeClass).fill(NodeClass.NoNode.id)
      val takes = where(nodeClass.elementEquals(noNode), Tensor.like(slot).fill(false), slot)
      def after[W](all: Tensor2[Drawing, Node, W], one: Tensor2[Drawing, Node, W]) = concatenate(all, one, Axis[Node])
      def placed(coordinate: Tensor2[Drawing, Node, Float32]) =
        where(takes, coordinate, Tensor.like(coordinate).fill(0f))
      val record = taken.copy(
        nodeClass = after(taken.nodeClass, where(takes, nodeClass, noNode)),
        startX = after(taken.startX, placed(startX)),
        startY = after(taken.startY, placed(startY)),
        endX = after(taken.endX, placed(endX)),
        endY = after(taken.endY, placed(endY))
      )
      (record, takes.squeeze(Axis[Node]))

    val (withNodes, _) = (0 until nodeSlots.size).foldLeft((nothingTaken, allTaking)):
      case ((taken, taking), _) => takeNode(taken, taking)

    /** The nodes of every record as the node decoder carries them, which is what an
      * [[EdgeDecoder]] relates, and which of the slots hold one. The prediction that door answers
      * with is thrown away, nothing being predicted there.
      */
    val (related, holds) =
      zipvmap(Axis[Drawing])(documents, withNodes.nodeClass, withNodes.startX, withNodes.startY, withNodes.endX, withNodes.endY):
        case (document, nodeClass, startX, startY, endX, endY) =>
          val nodes = RecordNodes(nodeClass, startX, startY, endX, endY)
          val carried = nodeDecoder.forTranscription(document, nodePosition.injectToPrefix(embedNodes(nodes)), params.nodes.token)._1
          (carried, holdsNode(nodes))

    /** One more relationship slot, taken the same way the nodes were. */
    def takeEdge(taken: RecordBatch[Drawing, Node, Edge], taking: Tensor1[Drawing, Bool]) =
      val (edgeClass, subject, obj) =
        zipvmap(Axis[Drawing])(documents, related, holds, taken.edgeClass, taken.subject, taken.obj):
          case (document, related, holds, edgeClass, subject, obj) =>
            nextEdge(document, related, holds, RecordEdges(edgeClass, subject, obj))
      val slot = taking.appendAxis(Axis[Edge])
      val noEdge = Tensor.like(edgeClass).fill(EdgeClass.NoEdge.id)
      val takes = where(edgeClass.elementEquals(noEdge), Tensor.like(slot).fill(false), slot)
      def after(all: Tensor2[Drawing, Edge, Int32], one: Tensor2[Drawing, Edge, Int32]) = concatenate(all, one, Axis[Edge])
      def named(end: Tensor2[Drawing, Edge, Int32]) = where(takes, end, Tensor.like(end).fill(0))
      val record = taken.copy(
        edgeClass = after(taken.edgeClass, where(takes, edgeClass, noEdge)),
        subject = after(taken.subject, named(subject)),
        obj = after(taken.obj, named(obj))
      )
      (record, takes.squeeze(Axis[Edge]))

    val (withEdges, _) = (0 until edgeSlots.size).foldLeft((withNodes, allTaking)):
      case ((taken, taking), _) => takeEdge(taken, taking)

    withEdges

  /** The node one drawing answers the slot after `taken` with, having taken the nodes before it. */
  private def nextNode(document: Tensor2[Patch, Embedding, V], taken: RecordNodes[Node]) =
    val at = taken.nodeClass.shape(Axis[Node])
    val context = concatenate(embedNodes(taken), params.nodes.token.prependAxis(Axis[Node]), Axis[Node])
    val placed = nodePosition.injectToPrefix(context)
    val (_, answered) = nodeDecoder.forTranscription(document, placed.slice(Axis[Node].at(0 until at)), placed.slice(Axis[Node].at(at)))
    val next = nodeScorer.decide(nodeScorer(answered.prependAxis(Axis[Node])))
    (next.nodeClass, next.startX, next.startY, next.endX, next.endY)

  /** The relationship it answers the slot after `taken` with, between the nodes `related` carries
    * and `holds` says are there.
    */
  private def nextEdge(
      document: Tensor2[Patch, Embedding, V],
      related: Tensor2[Node, Embedding, V],
      holds: Tensor1[Node, Bool],
      taken: RecordEdges[Edge]
  ) =
    val at = taken.edgeClass.shape(Axis[Edge])
    val context = concatenate(embedEdges(taken), params.edges.token.prependAxis(Axis[Edge]), Axis[Edge])
    val placed = edgePosition.injectToPrefix(context)
    val answered = edgeDecoder.forTranscription(document, related, holds, placed.slice(Axis[Edge].at(0 until at)), placed.slice(Axis[Edge].at(at)))
    val next = edgeScorer.decide(edgeScorer(answered.prependAxis(Axis[Edge])))
    (next.edgeClass, next.subject, next.obj)

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
      // A node embedding is put together from its class and the four coordinates a class can
      // place; a relationship embedding from its class and the two nodes it relates.
      val nodePartExtent = Axis[NodePart |*| PartEmbedding] -> 5 * partExtent.size
      val edgePartExtent = Axis[EdgePart |*| PartEmbedding] -> 3 * partExtent.size

      val (nodeDecoderKey, edgeDecoderKey) = decoderKey.splitToTuple(2)
      val (nodeEmbedderKey, edgeEmbedderKey) = embedderKey.splitToTuple(2)
      val (startXKey, startYKey, endXKey, endYKey, nodeClassKey, nodeProjectionKey) = nodeEmbedderKey.splitToTuple(6)
      val (subjectKey, objKey, edgeClassKey, edgeProjectionKey) = edgeEmbedderKey.splitToTuple(4)
      val (nodeHeadKey, edgeHeadKey) = scorerKey.splitToTuple(2)
      val (classHeadKey, startXHeadKey, startYHeadKey, endXHeadKey, endYHeadKey) = nodeHeadKey.splitToTuple(5)
      val (edgeClassHeadKey, subjectHeadKey, objHeadKey) = edgeHeadKey.splitToTuple(3)
      val (nodeTokenKey, edgeTokenKey) = tokenKey.splitToTuple(2)
      val (nodePositionKey, edgePositionKey) = positionKey.splitToTuple(2)

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
            startX = Init.xavierUniform(pixelExtent, partExtent, startXKey),
            startY = Init.xavierUniform(pixelExtent, partExtent, startYKey),
            endX = Init.xavierUniform(pixelExtent, partExtent, endXKey),
            endY = Init.xavierUniform(pixelExtent, partExtent, endYKey),
            projection = AffineLayer.Params.init(nodePartExtent, embeddingExtent, nodeProjectionKey)
          ),
          scorer = NodeScorer.Params(
            nodeClass = AffineLayer.Params.init(embeddingExtent, nodeClassExtent, classHeadKey),
            startX = AffineLayer.Params.init(embeddingExtent, pixelExtent, startXHeadKey),
            startY = AffineLayer.Params.init(embeddingExtent, pixelExtent, startYHeadKey),
            endX = AffineLayer.Params.init(embeddingExtent, pixelExtent, endXHeadKey),
            endY = AffineLayer.Params.init(embeddingExtent, pixelExtent, endYHeadKey)
          ),
          token = Init.xavierUniformVector(embeddingExtent, nodeTokenKey),
          positions = LearnedAbsolutePositionalInjector.Params.lecunNormal(nodeExtent, embeddingExtent, nodePositionKey)
        ),
        edges = EdgeParams(
          decoder = EdgeDecoder.Params.xavierUniformDepthScaled(numLayers, numHeads, embeddingExtent, embeddingExtent, embeddingMixedExtent, edgeDecoderKey),
          embedder = EdgeEmbedder.Params(
            edgeClass = Init.xavierUniform(edgeClassExtent, partExtent, edgeClassKey),
            subject = Init.xavierUniform(linkedExtent, partExtent, subjectKey),
            obj = Init.xavierUniform(linkedExtent, partExtent, objKey),
            projection = AffineLayer.Params.init(edgePartExtent, embeddingExtent, edgeProjectionKey)
          ),
          scorer = EdgeScorer.Params(
            edgeClass = AffineLayer.Params.init(embeddingExtent, edgeClassExtent, edgeClassHeadKey),
            subject = AffineLayer.Params.init(embeddingExtent, linkedExtent, subjectHeadKey),
            obj = AffineLayer.Params.init(embeddingExtent, linkedExtent, objHeadKey)
          ),
          token = Init.xavierUniformVector(embeddingExtent, edgeTokenKey),
          positions = LearnedAbsolutePositionalInjector.Params.lecunNormal(edgeExtent, embeddingExtent, edgePositionKey)
        )
      )
