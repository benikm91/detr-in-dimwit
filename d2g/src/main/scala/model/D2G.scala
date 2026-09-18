package d2g.model

import d2g.*
import d2g.train.*
import d2g.eval.*
import d2g.config.*
import dataset.EdgeClass
import dataset.NodeClass
import dataset.EdgeClasses
import dataset.NodeClasses
import dataset.Record
import dataset.RecordBatch
import dataset.RecordEdges
import dataset.RecordNodes
import deepwit.base.AffineLayer
import documentEncoder.DocumentEncoder
import deepwit.embedder.LearnedAbsolutePositionalInjector
import deepwit.init.Init
import EdgeScorer.EdgeLogits
import NodeScorer.NodeLogits
import dimwit.stats.Uniform
import dimwit.*
import dimwit.Conversions.given

import scala.language.implicitConversions

/** Document-to-graph model based on remaining-node prediction.
  *
  * 1. The document (or: image) is embedded by a vision transformer to a sequence of patch embeddings.
  * 2. The graph is predicted in two stages:
  *   a. the nodes based on cross-attenting the document (1)
  *   b. the relationships based on cross-attenting the document (1) and the nodes (2a).
  */
class D2G[V: IsFloating](params: D2G.Params[V]):

  import D2G.EdgeQueryLogits
  import D2G.NodeQueryLogits
  import D2G.Scores

  val encodeDocument = DocumentEncoder(Axis[Width], Axis[Height], Axis[Channel], params.encoder)

  private val embedNodes = NodeEmbedder(params.nodes.embedder)
  private val nodePosition = LearnedAbsolutePositionalInjector(params.nodes.positions)
  private val nodeDecoder = NodeDecoder(params.nodes.decoder)
  val nodeScorer = NodeScorer(params.nodes.scorer)

  private val embedEdges = EdgeEmbedder(params.edges.embedder)
  private val edgePosition = LearnedAbsolutePositionalInjector(params.edges.positions)
  private val edgeDecoder = EdgeDecoder(params.edges.decoder)
  val edgeScorer = EdgeScorer(params.edges.scorer)

  private val pool = params.nodes.queries.shape(Axis[Query])

  /** What two queries of the pool answer. Queries selected randomly. */
  def logits(document: Tensor3[Width, Height, Channel, V], taken: Record[Node, Edge], asked: Key): Scores[V] =
    val randomQueryIds = Random.permutation(Axis[Query] -> pool)(asked).slice(Axis[Query].at(0 until 2))
    predict(encodeDocument(document), taken, randomQueryIds, randomQueryIds)

  /** What every query of the pool answers, in one reading. */
  def logitsPerQuery(encoded: Tensor2[Patch, Embedding, V], taken: Record[Node, Edge]): Scores[V] =
    val allQueryIds = Tensor1(Axis[Query], VType[Int32]).fromArray(Array.range(0, pool))
    predict(encoded, taken, allQueryIds, allQueryIds)

  /** What each asked query answers at every slot. */
  private def predict(
      encodedDocument: Tensor2[Patch, Embedding, V],
      taken: Record[Node, Edge],
      nodeQueryIds: Tensor1[Query, Int32],
      edgeQueryIds: Tensor1[Query, Int32]
  ): Scores[V] =

    val (carriedNodes, answeredNodes) =
      val takenNodes = nodePosition(embedNodes(taken.nodes))
      val queryNodes = params.nodes.queries.take(Axis[Query])(nodeQueryIds) // take queries and broadcast along context
        .vmap(Axis[Query]): query =>
          nodePosition(query.broadcastTo(takenNodes.shape))
      nodeDecoder.forTraining(encodedDocument, takenNodes, queryNodes)

    val (nodeClass, startX, startY, endX, endY) =
      answeredNodes.vmap(Axis[Query]): answered =>
        val scored = nodeScorer(answered)
        (scored.nodeClass, scored.startX, scored.startY, scored.endX, scored.endY)

    val (_, answeredEdges) =
      val nodeSource =
        val nodesPresentMask = !(taken.nodes.nodeClass elementEquals_! NodeClass.NoNode.id)
        NodeSource(carriedNodes, nodesPresentMask)
      val takenEdges = edgePosition(embedEdges(taken.edges))
      val queryEdges = params.edges.queries.take(Axis[Query])(edgeQueryIds) // take queries and broadcast along context
        .vmap(Axis[Query]): query =>
          edgePosition(query.broadcastTo(takenEdges.shape))
      edgeDecoder.forTraining(encodedDocument, nodeSource, takenEdges, queryEdges)

    val (edgeClass, subject, obj) =
      answeredEdges.vmap(Axis[Query]): answered =>
        val scored = edgeScorer(answered)
        (scored.edgeClass, scored.subject, scored.obj)

    Scores(NodeQueryLogits(nodeClass, startX, startY, endX, endY), EdgeQueryLogits(edgeClass, subject, obj))

object D2G:

  case class Scores[V](nodes: NodeQueryLogits[V], edges: EdgeQueryLogits[V])

  /** [[NodeLogits]] at every query slot. */
  case class NodeQueryLogits[V](
      nodeClass: Tensor3[Query, Node, NodeClasses, V],
      startX: Tensor3[Query, Node, Pixel, V],
      startY: Tensor3[Query, Node, Pixel, V],
      endX: Tensor3[Query, Node, Pixel, V],
      endY: Tensor3[Query, Node, Pixel, V]
  ):
    def at(query: Int): NodeLogits[V] = NodeLogits(
      nodeClass.slice(Axis[Query].at(query)),
      startX.slice(Axis[Query].at(query)),
      startY.slice(Axis[Query].at(query)),
      endX.slice(Axis[Query].at(query)),
      endY.slice(Axis[Query].at(query))
    )

  object NodeQueryLogits:

    def of[V](answered: Seq[NodeLogits[V]]): NodeQueryLogits[V] = NodeQueryLogits(
      nodeClass = stack(answered.map(_.nodeClass), Axis[Query]),
      startX = stack(answered.map(_.startX), Axis[Query]),
      startY = stack(answered.map(_.startY), Axis[Query]),
      endX = stack(answered.map(_.endX), Axis[Query]),
      endY = stack(answered.map(_.endY), Axis[Query])
    )

  /** [[EdgeLogits]] at every query slot. */
  case class EdgeQueryLogits[V](
      edgeClass: Tensor3[Query, Edge, EdgeClasses, V],
      subject: Tensor3[Query, Edge, LinkedNode, V],
      obj: Tensor3[Query, Edge, LinkedNode, V]
  ):
    def at(query: Int): EdgeLogits[V] = EdgeLogits(
      edgeClass.slice(Axis[Query].at(query)),
      subject.slice(Axis[Query].at(query)),
      obj.slice(Axis[Query].at(query))
    )

  object EdgeQueryLogits:

    def of[V](answered: Seq[EdgeLogits[V]]): EdgeQueryLogits[V] = EdgeQueryLogits(
      edgeClass = stack(answered.map(_.edgeClass), Axis[Query]),
      subject = stack(answered.map(_.subject), Axis[Query]),
      obj = stack(answered.map(_.obj), Axis[Query])
    )

  case class Params[V](
      encoder: DocumentEncoder.Params[Embedding, V],
      nodes: Params.NodeParams[V],
      edges: Params.EdgeParams[V]
  )

  object Params:

    given tensorTree: TensorTree[Params[Float32]] = TensorTree.derived
    given tree: TreeOf[Params[Float32], Float32] = TreeOf.derived

    case class NodeParams[V](
        decoder: NodeDecoder.Params[Embedding, Embedding, V],
        embedder: NodeEmbedder.Params[V],
        scorer: NodeScorer.Params[V],
        queries: Tensor2[Query, Embedding, V],
        positions: LearnedAbsolutePositionalInjector.Params[Node, Embedding, V]
    )

    // Each half derives its own tree: derived for the whole at once, the derivation outgrows the
    // 64 KB the JVM allows one method.
    object NodeParams:

      given tensorTree: TensorTree[NodeParams[Float32]] = TensorTree.derived
      given tree: TreeOf[NodeParams[Float32], Float32] = TreeOf.derived

    case class EdgeParams[V](
        decoder: EdgeDecoder.Params[Embedding, Embedding, V],
        embedder: EdgeEmbedder.Params[V],
        scorer: EdgeScorer.Params[V],
        queries: Tensor2[Query, Embedding, V],
        positions: LearnedAbsolutePositionalInjector.Params[Edge, Embedding, V]
    )

    object EdgeParams:

      given tensorTree: TensorTree[EdgeParams[Float32]] = TensorTree.derived
      given tree: TreeOf[EdgeParams[Float32], Float32] = TreeOf.derived

    /** @param nodes  How many nodes of a record the model can hold. One more than the most any
      *               record of the data draws, so that the last prediction embedding has somewhere
      *               to say the nodes have ended.
      * @param edges  The same for the relationships between them.
      * @param queries How many query vectors a slot may be asked with — the pool.
      * @param canvas The width of the drawing, which is how many pixels a coordinate chooses from.
      */
    def init(
        numLayers: Int,
        numHeads: Int,
        embedding: Int,
        nodes: Int,
        edges: Int,
        queries: Int,
        canvas: Int,
        key: Key
    ): Params[Float32] =
      val (encoderKey, decoderKey, embedderKey, scorerKey, tokenKey, positionKey) = key.splitToTuple(6)

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
        encoder = DocumentEncoder.Params.xavierUniformDepthScaled(numLayers, numHeads, embeddingExtent, Axis[DocumentEncoder.EmbeddingMixed] -> embeddingMixedExtent.size, encoderKey),
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
          queries = Init.xavierUniform(Axis[Query] -> queries, embeddingExtent, nodeTokenKey),
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
          queries = Init.xavierUniform(Axis[Query] -> queries, embeddingExtent, edgeTokenKey),
          positions = LearnedAbsolutePositionalInjector.Params.lecunNormal(edgeExtent, embeddingExtent, edgePositionKey)
        )
      )
