package d2g.sketch.model

import d2g.*
import d2g.model.*
import d2s.model.*
import d2s.sketch.model.D2S
import d2s.sketch.model.D2S.NodeQueryLogits
import dataset.EdgeClass
import dataset.EdgeClasses
import dataset.NodeClass
import dataset.Record
import dataset.RecordEdges
import deepwit.attention.KVCache
import deepwit.base.AffineLayer
import deepwit.embedder.LearnedAbsolutePositionalInjector
import deepwit.embedder.VocabularyEmbedder
import deepwit.init.Init
import EdgeHead.EdgeLogits
import dimwit.*

import scala.language.implicitConversions

/** Document-to-graph model: a [[D2S]] for the nodes of a drawing, and the relationships between
  * them on top, as [[d2g.floorplan.model.D2G]] writes those of a floor plan.
  */
class D2G[V: IsFloating](params: D2G.Params[V]):

  import D2G.EdgeQueryLogits
  import D2G.Scores

  /** The model of the nodes, which the relationships are written on top of. */
  val set = D2S(params.set)
  val encodeDocument = set.encodeDocument

  private val embedEdges = EdgeEmbedder(params.edges.embedder)
  private val edgePosition = LearnedAbsolutePositionalInjector(params.edges.positions)
  private val edgeDecoder = EdgeDecoder(params.edges.decoder)
  val edgeHead = EdgeHead(params.edges.head)

  /** What two queries of the pool answer. Queries selected randomly. */
  def logits(document: Tensor3[Width, Height, Channel, V], taken: Record[Node, Edge], asked: Key): Scores[V] =
    val randomQueryIds = Random.permutation(Axis[PoolQuery] -> set.pool)(asked).slice(Axis[PoolQuery].at(0 until 2))
    val encodedDocument = encodeDocument(document)

    val (carriedNodes, nodes) = set.predict(encodedDocument, taken.nodes, randomQueryIds)

    val nodeSource = NodeSource(carriedNodes, !(taken.nodeClass elementEquals_! NodeClass.NoNode.id))
    val (_, answeredEdges) =
      val takenEdges = edgePosition(embedEdges(taken.edges, carriedNodes))
      val queryEdges = params.edges.queries.slice(Axis[PoolQuery].at(randomQueryIds))
        .vmap(Axis[PoolQuery])(query => edgePosition(query.broadcastTo(takenEdges.shape)))
      edgeDecoder.forTraining(encodedDocument, nodeSource, takenEdges, queryEdges)

    val (edgeClass, subject, obj) =
      answeredEdges.vmap(Axis[PoolQuery]): answered =>
        val scored = edgeHead(answered, nodeSource)
        (scored.edgeClass, scored.subject, scored.obj)

    Scores(nodes, EdgeQueryLogits(edgeClass, subject, obj))

  /** Every edge decoder block's keys and values of the encoded document and of the written `nodes`,
    * which every relationship step reads.
    */
  def readForEdges(encodedDocument: Tensor2[Patch, Embedding, V], nodes: NodeSource[Embedding, V]): List[EdgeSources[V]] =
    edgeDecoder.read(encodedDocument, nodes)

  /** The edge decoder before any relationship is taken. */
  def noEdgeTaken(slots: AxisExtent[Edge]): List[KVCache[Edge, V]] =
    edgeDecoder.nothingTaken(slots)

  /** Takes `edge`, a record of one relationship between `nodes`, at `slot`, for the slots after it to read. */
  def takeEdge(nodes: NodeSource[Embedding, V], sources: List[EdgeSources[V]], taken: List[KVCache[Edge, V]], slot: Tensor0[Int32], edge: RecordEdges[Edge]): List[KVCache[Edge, V]] =
    edgeDecoder.take(sources, taken, slot, edgePosition.injectAt(slot)(embedEdges(edge, nodes.nodeEmbeddings).squeeze(Axis[Edge])))

  /** What the pool answers at relationship `slot`, given the relationships taken before it: a
    * candidate relationship between `nodes` per query, in the order of the queries.
    */
  def answerEdgeAt(nodes: NodeSource[Embedding, V], sources: List[EdgeSources[V]], taken: List[KVCache[Edge, V]], slot: Tensor0[Int32]): EdgeLogits[V] =
    val asked = params.edges.queries.vmap(Axis[PoolQuery])(edgePosition.injectAt(slot))
    edgeHead(edgeDecoder.answer(sources, taken, slot, asked).relabel(Axis[PoolQuery] -> Axis[Edge]), nodes)

object D2G:

  case class Scores[V](nodes: NodeQueryLogits[V], edges: EdgeQueryLogits[V])

  object Scores:

    given tensorTree[V]: TensorTree[Scores[V]] = TensorTree.derived
    given tree[V]: TreeOf[Scores[V], V] = TreeOf.derived

  /** [[EdgeLogits]] at every query slot. */
  case class EdgeQueryLogits[V](
      edgeClass: Tensor3[PoolQuery, Edge, EdgeClasses, V],
      subject: Tensor3[PoolQuery, Edge, Node, V],
      obj: Tensor3[PoolQuery, Edge, Node, V]
  ):
    def at(query: Int): EdgeLogits[V] = EdgeLogits(
      edgeClass.slice(Axis[PoolQuery].at(query)),
      subject.slice(Axis[PoolQuery].at(query)),
      obj.slice(Axis[PoolQuery].at(query))
    )

  object EdgeQueryLogits:

    given tensorTree[V]: TensorTree[EdgeQueryLogits[V]] = TensorTree.derived
    given tree[V]: TreeOf[EdgeQueryLogits[V], V] = TreeOf.derived

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

    /** The relationships, drawn fresh, on top of `set`, a trained model of the nodes.
      *
      * @param edges   How many relationships of a record the model can hold. One more than the most
      *                any record of the data holds, so that the last prediction embedding has
      *                somewhere to say the relationships have ended.
      * @param queries How many query vectors a slot may be asked with — the pool.
      */
    def init(set: D2S.Params[Float32], numLayers: Int, numHeads: Int, embedding: Int, edges: Int, queries: Int, key: Key): Params[Float32] =
      val (edgeDecoderKey, edgeEmbedderKey, edgeHeadKey, edgeTokenKey, edgePositionKey) = key.splitToTuple(5)

      val embeddingExtent = Axis[Embedding] -> embedding
      val embeddingMixedExtent = Axis[EmbeddingMixed] -> embedding * 4
      val edgeClassExtent = Axis[EdgeClasses] -> EdgeClass.values.length

      val (subjectKey, objKey, edgeClassKey) = edgeEmbedderKey.splitToTuple(3)
      val (edgeClassHeadKey, subjectHeadKey, objHeadKey) = edgeHeadKey.splitToTuple(3)

      Params(
        set = set,
        edges = EdgeParams(
          decoder = EdgeDecoder.Params.xavierUniformDepthScaled(numLayers, numHeads, embeddingExtent, embeddingExtent, embeddingMixedExtent, edgeDecoderKey),
          embedder = EdgeEmbedder.Params(
            edgeClass = VocabularyEmbedder.Params.init(edgeClassExtent, embeddingExtent, edgeClassKey),
            subject = AffineLayer.Params.init(embeddingExtent, embeddingExtent, subjectKey),
            obj = AffineLayer.Params.init(embeddingExtent, embeddingExtent, objKey)
          ),
          head = EdgeHead.Params(
            edgeClass = AffineLayer.Params.init(embeddingExtent, edgeClassExtent, edgeClassHeadKey),
            subject = Pointer.Params.init(embeddingExtent, subjectHeadKey),
            obj = Pointer.Params.init(embeddingExtent, objHeadKey)
          ),
          queries = Init.xavierUniform(Axis[PoolQuery] -> queries, embeddingExtent, edgeTokenKey),
          positions = LearnedAbsolutePositionalInjector.Params.lecunNormal(Axis[Edge] -> edges, embeddingExtent, edgePositionKey)
        )
      )
