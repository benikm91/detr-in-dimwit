package d2g.model

import d2g.*
import d2g.train.*
import d2g.eval.*
import d2g.config.*
import d2s.model.*
import d2s.model.D2S.NodeQueryLogits
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
import deepwit.embedder.VocabularyEmbedder
import deepwit.init.Init
import EdgeHead.EdgeLogits
import NodeHead.NodeLogits
import dimwit.stats.Uniform
import dimwit.*

import scala.language.implicitConversions

/** Document-to-graph model: a [[D2S]] for the nodes, and the relationships on top.
  *
  * 1. The document (or: image) is embedded by a vision transformer to a sequence of patch embeddings.
  * 2. The graph is predicted in two stages:
  *   a. the nodes based on cross-attenting the document (1), by the [[D2S]]
  *   b. the relationships based on cross-attenting the document (1) and the nodes (2a).
  */
class D2G[V: IsFloating](params: D2G.Params[V]):

  import D2G.EdgeQueryLogits
  import D2G.Scores

  private val set = D2S(params.set)
  val encodeDocument = set.encodeDocument
  val nodeHead = set.nodeHead

  private val embedEdges = EdgeEmbedder(params.edges.embedder)
  private val edgePosition = LearnedAbsolutePositionalInjector(params.edges.positions)
  private val edgeDecoder = EdgeDecoder(params.edges.decoder)
  val edgeHead = EdgeHead(params.edges.head)

  private val pool = set.pool

  /** What two queries of the pool answer. Queries selected randomly. */
  def logits(document: Tensor3[Width, Height, Channel, V], taken: Record[Node, Edge], asked: Key): Scores[V] =
    val randomQueryIds = Random.permutation(Axis[PoolQuery] -> pool)(asked).slice(Axis[PoolQuery].at(0 until 2))
    predict(encodeDocument(document), taken, randomQueryIds, randomQueryIds)

  /** What every query of the pool answers, in one reading. */
  def logitsPerQuery(encoded: Tensor2[Patch, Embedding, V], taken: Record[Node, Edge]): Scores[V] =
    val allQueryIds = Tensor1(Axis[PoolQuery], VType[Int32]).fromArray(Array.range(0, pool))
    predict(encoded, taken, allQueryIds, allQueryIds)

  /** What each asked query answers at every slot. */
  private def predict(
      encodedDocument: Tensor2[Patch, Embedding, V],
      taken: Record[Node, Edge],
      nodeQueryIds: Tensor1[PoolQuery, Int32],
      edgeQueryIds: Tensor1[PoolQuery, Int32]
  ): Scores[V] =

    val (carriedNodes, nodes) = set.predict(encodedDocument, taken.nodes, nodeQueryIds)

    val (_, answeredEdges) =
      val nodeSource =
        val nodesPresentMask = !(taken.nodes.nodeClass elementEquals_! NodeClass.NoNode.id)
        NodeSource(carriedNodes, nodesPresentMask)
      val takenEdges = edgePosition(embedEdges(taken.edges))
      val queryEdges = params.edges.queries.take(Axis[PoolQuery])(edgeQueryIds) // take queries and broadcast along context
        .vmap(Axis[PoolQuery]): query =>
          edgePosition(query.broadcastTo(takenEdges.shape))
      edgeDecoder.forTraining(encodedDocument, nodeSource, takenEdges, queryEdges)

    val (edgeClass, subject, obj) =
      answeredEdges.vmap(Axis[PoolQuery]): answered =>
        val scored = edgeHead(answered)
        (scored.edgeClass, scored.subject, scored.obj)

    Scores(nodes, EdgeQueryLogits(edgeClass, subject, obj))

object D2G:

  case class Scores[V](nodes: NodeQueryLogits[V], edges: EdgeQueryLogits[V])

  object Scores:

    given tensorTree[V]: TensorTree[Scores[V]] = TensorTree.derived
    given tree[V]: TreeOf[Scores[V], V] = TreeOf.derived

  /** [[EdgeLogits]] at every query slot. */
  case class EdgeQueryLogits[V](
      edgeClass: Tensor3[PoolQuery, Edge, EdgeClasses, V],
      subject: Tensor3[PoolQuery, Edge, LinkedNode, V],
      obj: Tensor3[PoolQuery, Edge, LinkedNode, V]
  ):
    def at(query: Int): EdgeLogits[V] = EdgeLogits(
      edgeClass.slice(Axis[PoolQuery].at(query)),
      subject.slice(Axis[PoolQuery].at(query)),
      obj.slice(Axis[PoolQuery].at(query))
    )

  object EdgeQueryLogits:

    given tensorTree[V]: TensorTree[EdgeQueryLogits[V]] = TensorTree.derived
    given tree[V]: TreeOf[EdgeQueryLogits[V], V] = TreeOf.derived

    def of[V](answered: Seq[EdgeLogits[V]]): EdgeQueryLogits[V] = EdgeQueryLogits(
      edgeClass = stack(answered.map(_.edgeClass), Axis[PoolQuery]),
      subject = stack(answered.map(_.subject), Axis[PoolQuery]),
      obj = stack(answered.map(_.obj), Axis[PoolQuery])
    )

  case class Params[V](
      set: D2S.Params[V],
      edges: Params.EdgeParams[V]
  )

  object Params:

    given tensorTree: TensorTree[Params[Float32]] = TensorTree.derived
    given tree: TreeOf[Params[Float32], Float32] = TreeOf.derived

    case class EdgeParams[V](
        decoder: EdgeDecoder.Params[Embedding, Embedding, V],
        embedder: EdgeEmbedder.Params[V],
        head: EdgeHead.Params[V],
        queries: Tensor2[PoolQuery, Embedding, V],
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
      val (setKey, edgeDecoderKey, edgeEmbedderKey, edgeHeadKey, edgeTokenKey, edgePositionKey) = key.splitToTuple(6)

      val embeddingExtent = Axis[Embedding] -> embedding
      val embeddingMixedExtent = Axis[EmbeddingMixed] -> embedding * 4
      val partExtent = Axis[PartEmbedding] -> embedding / 8
      val edgeClassExtent = Axis[dataset.EdgeClasses] -> EdgeClass.values.length
      val edgeExtent = Axis[Edge] -> edges
      val linkedExtent = Axis[LinkedNode] -> nodes
      // A relationship embedding is put together from its class and the two nodes it relates.
      val edgePartExtent = Axis[EdgePart |*| PartEmbedding] -> 3 * partExtent.size

      val (subjectKey, objKey, edgeClassKey, edgeProjectionKey) = edgeEmbedderKey.splitToTuple(4)
      val (edgeClassHeadKey, subjectHeadKey, objHeadKey) = edgeHeadKey.splitToTuple(3)

      Params(
        set = D2S.Params.init(numLayers, numHeads, embedding, nodes, queries, canvas, setKey),
        edges = EdgeParams(
          decoder = EdgeDecoder.Params.xavierUniformDepthScaled(numLayers, numHeads, embeddingExtent, embeddingExtent, embeddingMixedExtent, edgeDecoderKey),
          embedder = EdgeEmbedder.Params(
            edgeClass = VocabularyEmbedder.Params.init(edgeClassExtent, partExtent, edgeClassKey),
            subject = VocabularyEmbedder.Params.init(linkedExtent, partExtent, subjectKey),
            obj = VocabularyEmbedder.Params.init(linkedExtent, partExtent, objKey),
            projection = AffineLayer.Params.init(edgePartExtent, embeddingExtent, edgeProjectionKey)
          ),
          head = EdgeHead.Params(
            edgeClass = AffineLayer.Params.init(embeddingExtent, edgeClassExtent, edgeClassHeadKey),
            subject = AffineLayer.Params.init(embeddingExtent, linkedExtent, subjectHeadKey),
            obj = AffineLayer.Params.init(embeddingExtent, linkedExtent, objHeadKey)
          ),
          queries = Init.xavierUniform(Axis[PoolQuery] -> queries, embeddingExtent, edgeTokenKey),
          positions = LearnedAbsolutePositionalInjector.Params.lecunNormal(edgeExtent, embeddingExtent, edgePositionKey)
        )
      )
