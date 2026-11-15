package d2gtok.model

import d2gtok.*
import d2g.floorplan.model.D2G
import d2g.floorplan.model.D2G.EdgeQueryLogits
import d2g.floorplan.model.EdgeEmbedder
import d2g.floorplan.model.EdgeHead
import d2g.floorplan.model.EdgeHead.EdgeLogits
import d2g.model.EdgeDecoder
import d2g.model.EdgeSources
import d2g.model.NodeSource
import d2s.model.PoolQuery
import d2tok.model.D2Tok
import deepwit.attention.KVCache
import deepwit.embedder.LearnedAbsolutePositionalInjector
import dimwit.*
import syntheticFloorplansDataset.FloorplanEdges

import scala.language.implicitConversions

/** Document-to-graph model on the nodes of a [[D2Tok]]: d2tok writes a floor plan's nodes in its
  * words, and the relationship decoder of [[D2G]] writes the relationships on top of them.
  *
  * The relationships read a node, and point at it, as the d2tok decoder carried the position after
  * its last word: under causal attention, that position has read every word of the node.
  */
class D2GTok[V: IsFloating](params: D2GTok.Params[V]):

  /** The model of the nodes, whose encoding of the document the relationships read too. */
  val nodes = D2Tok(params.nodes)

  private val embedEdges = EdgeEmbedder(params.edges.embedder)
  private val edgePosition = LearnedAbsolutePositionalInjector(params.edges.positions)
  private val edgeDecoder = EdgeDecoder(params.edges.decoder)
  val edgeHead = EdgeHead(params.edges.head)

  /** The scores of every node word, read from the words `read` up to it; and what two queries of the
    * pool, selected randomly, answer at every relationship slot, given the relationships `taken`
    * before it between the nodes `present`.
    */
  def logits(
      document: Tensor3[Width, Height, Channel, V],
      read: Tensor1[Token, Int32],
      present: Tensor1[Node, Bool],
      taken: FloorplanEdges[Edge],
      asked: Key
  ): (Tensor2[Token, Word, V], EdgeQueryLogits[V]) =
    val encoded = nodes.encodeDocument(document)
    val carried = nodes.carried(encoded, read)
    val written = nodesOf(carried, present)

    val pool = Axis[PoolQuery] -> params.edges.queries.shape(Axis[PoolQuery])
    val askedQueries = Random.permutation(pool)(asked).slice(Axis[PoolQuery].at(0 until 2))
    val takenEdges = edgePosition(embedEdges(taken, written.nodeEmbeddings))
    val queryEdges = params.edges.queries.slice(Axis[PoolQuery].at(askedQueries))
      .vmap(Axis[PoolQuery])(query => edgePosition(query.broadcastTo(takenEdges.shape)))
    val (_, answered) = edgeDecoder.forTraining(asEdgeEmbedding(encoded), written, takenEdges, queryEdges)
    val (relationship, subject, obj) =
      answered.vmap(Axis[PoolQuery]): answering =>
        val scored = edgeHead(answering, written)
        (scored.relationship, scored.subject, scored.obj)

    (nodes.scored(carried), EdgeQueryLogits(relationship, subject, obj))

  /** The nodes as the relationships read them: each as d2tok `carried` the position after its last
    * word, and which of them are there.
    */
  def nodesOf(carried: Tensor2[Token, d2tok.Embedding, V], present: Tensor1[Node, Bool]): NodeSource[Embedding, V] =
    val slots = present.shape.extent(Axis[Node])
    // The words of every node, then the end.
    val wordsPerNode = (carried.shape(Axis[Token]) - 1) / slots.size
    val afterEachNode = carried
      .slice(Axis[Token].at(1 to slots.size * wordsPerNode))
      .relabel(Axis[Token] -> Axis[Node |*| d2tok.Part])
      .unflatten(Axis[Node |*| d2tok.Part], Shape(slots, Axis[d2tok.Part] -> wordsPerNode))
      .slice(Axis[d2tok.Part].at(wordsPerNode - 1))
    NodeSource(asEdgeEmbedding(afterEachNode), present)

  /** What d2tok embeds, read by the relationships as an embedding of their own. */
  private def asEdgeEmbedding[Row: Label](embedded: Tensor2[Row, d2tok.Embedding, V]): Tensor2[Row, Embedding, V] =
    embedded.relabel(Axis[d2tok.Embedding] -> Axis[Embedding])

  // Writing the relationships down one at a time, as [[D2G]] does.

  /** Every edge decoder block's keys and values of the `encoded` document and of the written
    * `nodes`, which every relationship step reads.
    */
  def readForEdges(encoded: Tensor2[Patch, d2tok.Embedding, V], nodes: NodeSource[Embedding, V]): List[EdgeSources[V]] =
    edgeDecoder.read(asEdgeEmbedding(encoded), nodes)

  /** The edge decoder before any relationship is taken. */
  def noEdgeTaken(slots: AxisExtent[Edge]): List[KVCache[Edge, V]] =
    edgeDecoder.nothingTaken(slots)

  /** Takes `edge`, a record of one relationship between `nodes`, at `slot`, for the slots after it to read. */
  def takeEdge(nodes: NodeSource[Embedding, V], sources: List[EdgeSources[V]], taken: List[KVCache[Edge, V]], slot: Tensor0[Int32], edge: FloorplanEdges[Edge]): List[KVCache[Edge, V]] =
    edgeDecoder.take(sources, taken, slot, edgePosition.injectAt(slot)(embedEdges(edge, nodes.nodeEmbeddings).squeeze(Axis[Edge])))

  /** What the pool answers at relationship `slot`, given the relationships taken before it: a
    * candidate relationship between `nodes` per query, in the order of the queries.
    */
  def answerEdgeAt(nodes: NodeSource[Embedding, V], sources: List[EdgeSources[V]], taken: List[KVCache[Edge, V]], slot: Tensor0[Int32]): EdgeLogits[V] =
    val asked = params.edges.queries.vmap(Axis[PoolQuery])(edgePosition.injectAt(slot))
    edgeHead(edgeDecoder.answer(sources, taken, slot, asked).relabel(Axis[PoolQuery] -> Axis[Edge]), nodes)

object D2GTok:

  case class Params[V](
      nodes: D2Tok.Params[V],
      edges: D2G.Params.EdgeParams[V]
  )

  object Params:

    given tensorTree: TensorTree[Params[Float32]] = TensorTree.derived
    given tree: TreeOf[Params[Float32], Float32] = TreeOf.derived

    /** The relationships of a [[D2G]], drawn as d2g draws them, on top of the trained `nodes`. The
      * d2s that d2g would write the nodes with is drawn too, and dropped.
      *
      * @param nodeSlots How many nodes of a record d2tok writes.
      * @param edgeSlots How many relationships of a record the model can hold, one more than the
      *                  most any record of the data holds, for the end.
      * @param queries   How many query vectors a relationship slot may be asked with — the pool.
      */
    def init(
        nodes: D2Tok.Params[Float32],
        numLayers: Int,
        numHeads: Int,
        embedding: Int,
        nodeSlots: Int,
        edgeSlots: Int,
        queries: Int,
        canvas: Int,
        key: Key
    ): Params[Float32] =
      Params(nodes, D2G.Params.init(numLayers, numHeads, embedding, nodeSlots, edgeSlots, queries, canvas, key).edges)
