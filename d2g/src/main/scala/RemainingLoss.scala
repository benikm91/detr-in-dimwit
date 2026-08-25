import dataset.EdgeClass
import dataset.EdgeClasses
import dataset.NodeClass
import dataset.NodeClasses
import dataset.RecordEdges
import dataset.RecordNodes
import EdgeScorer.EdgeLogits
import NodeScorer.NodeLogits
import dimwit.*
import dimwit.Conversions.given

import scala.language.implicitConversions

/** Equation 4 of the paper over the nodes of a record: the loss of remaining-node prediction.
  *
  * A prediction embedding may answer with any node the slots before it have not taken, so its
  * cost is the smallest dissimilarity to any of them rather than the dissimilarity to one.
  *
  * Two terms complete it: the empty slot just past the last node, so that transcription knows
  * where to stop, and the pass-through of every taken node, which is what keeps a node embedding
  * carrying its node while the prediction embedding beside it becomes another one.
  */
class RemainingNodeLoss[V: IsFloating](vtype: VType[V], canvas: Int) extends ((D2G.NodeScores[V], RecordNodes[Node]) => Tensor0[V]):

  /** Axis of the record's nodes seen as candidates to answer with rather than as positions. */
  private type Candidate = Prime[Node]

  override def apply(scored: D2G.NodeScores[V], target: RecordNodes[Node]): Tensor0[V] =
    val nodes = target.nodeClass.shape.extent(Axis[Node])
    val pairs = Shape2(nodes, Axis[Candidate] -> nodes.size)
    val holdsNode = NodeClass.indicator(vtype)(_.isDrawn).take(Axis[NodeClasses])(target.nodeClass)
    val taken = holdsNode.sum
    val candidates = triu(Tensor(pairs, vtype).fill(1f)) * holdsNode.relabelTo(Axis[Candidate]).broadcastTo(pairs)

    val remaining = cheapest(dissimilarity(scored.remaining, target), candidates) * candidates.max(Axis[Candidate])
    val stops = costOfClass(scored.remaining.nodeClass, NodeClass.NoNode.id) * isAt(taken, nodes, vtype)
    val passedThrough = dissimilarity(scored.taken, target) * Tensor2.eye(nodes, vtype) * holdsNode.broadcastTo(pairs)

    (remaining.sum + stops.sum) / (taken + Tensor0(vtype)(1f)) +
      passedThrough.sum / maximum(taken, Tensor0(vtype)(1f))

  /** What every position's scores would cost against every node of the record: its class, and
    * where the *target* node is placed — so that nothing depends on what the model predicts. A
    * class that runs nowhere is not measured on where it ends.
    */
  private def dissimilarity(logits: NodeLogits[V], target: RecordNodes[Node]): Tensor2[Node, Candidate, V] =
    val candidateClass = target.nodeClass.relabelTo(Axis[Candidate])
    def placed(scores: Tensor2[Node, Pixel, V], coordinate: Tensor1[Node, Float32]) =
      costOfValue(scores, Pixels.of(coordinate, canvas).relabelTo(Axis[Candidate]))
    val runsOn = NodeClass.indicator(vtype)(_.numPoints > 1).take(Axis[NodeClasses])(candidateClass)
    val ends = placed(logits.endX, target.endX) + placed(logits.endY, target.endY)
    costOfValue(logits.nodeClass, candidateClass) +
      placed(logits.startX, target.startX) +
      placed(logits.startY, target.startY) +
      ends * runsOn.broadcastTo(ends.shape)

/** The same over the relationships of a record, which are predicted the same way and cost the
  * same three terms — a relationship carries the nodes it links where a node carries its points.
  */
class RemainingEdgeLoss[V: IsFloating](vtype: VType[V]) extends ((D2G.EdgeScores[V], RecordEdges[Edge]) => Tensor0[V]):

  /** Axis of the record's relationships seen as candidates to answer with rather than as
    * positions.
    */
  private type Candidate = Prime[Edge]

  override def apply(scored: D2G.EdgeScores[V], target: RecordEdges[Edge]): Tensor0[V] =
    val edges = target.edgeClass.shape.extent(Axis[Edge])
    val pairs = Shape2(edges, Axis[Candidate] -> edges.size)
    val holdsEdge = EdgeClass.indicator(vtype)(_.relates).take(Axis[EdgeClasses])(target.edgeClass)
    val taken = holdsEdge.sum
    val candidates = triu(Tensor(pairs, vtype).fill(1f)) * holdsEdge.relabelTo(Axis[Candidate]).broadcastTo(pairs)

    val remaining = cheapest(dissimilarity(scored.remaining, target), candidates) * candidates.max(Axis[Candidate])
    val stops = costOfClass(scored.remaining.edgeClass, EdgeClass.NoEdge.id) * isAt(taken, edges, vtype)
    val passedThrough = dissimilarity(scored.taken, target) * Tensor2.eye(edges, vtype) * holdsEdge.broadcastTo(pairs)

    (remaining.sum + stops.sum) / (taken + Tensor0(vtype)(1f)) +
      passedThrough.sum / maximum(taken, Tensor0(vtype)(1f))

  /** The same for a relationship, which carries the two nodes it relates where a node carries the
    * points it is placed by. Only relationships are ever candidates, so both ends always count.
    */
  private def dissimilarity(logits: EdgeLogits[V], target: RecordEdges[Edge]): Tensor2[Edge, Candidate, V] =
    def named(scores: Tensor2[Edge, LinkedNode, V], end: Tensor1[Edge, Int32]) =
      costOfValue(scores, end.relabelTo(Axis[Candidate]))
    costOfValue(logits.edgeClass, target.edgeClass.relabelTo(Axis[Candidate])) +
      named(logits.subject, target.subject) +
      named(logits.obj, target.obj)

/** The smallest cost among the candidates of a position. The others are lifted above the whole
  * matrix rather than dropped, so that the minimum stays a plain reduction.
  */
private def cheapest[Slot: Label, Candidate: Label, V: IsFloating](
    cost: Tensor2[Slot, Candidate, V],
    candidates: Tensor2[Slot, Candidate, V]
): Tensor1[Slot, V] =
  val beyond = cost.max - cost.min + Tensor0(cost.vtype)(1f)
  where(candidates > Tensor.like(candidates).fill(0f), cost, cost + beyond.broadcastTo(cost.shape)).min(Axis[Candidate])

/** The cross entropy of every position's scores against the value every candidate holds. */
private def costOfValue[Slot: Label, Candidate: Label, L: Label, V: IsFloating](
    logits: Tensor2[Slot, L, V],
    values: Tensor1[Candidate, Int32]
): Tensor2[Slot, Candidate, V] =
  val chosen = logits.take(Axis[L])(values)
  logNormalizer(logits).broadcastTo(chosen.shape) - chosen

/** The cross entropy of every position's scores against one class, which is the one that ends the
  * record.
  */
private def costOfClass[Slot: Label, Classes: Label, V: IsFloating](logits: Tensor2[Slot, Classes, V], id: Int): Tensor1[Slot, V] =
  logNormalizer(logits) - logits.slice(Axis[Classes].at(id))

private def logNormalizer[Slot: Label, L: Label, V: IsFloating](logits: Tensor2[Slot, L, V]): Tensor1[Slot, V] =
  val peak = logits.max(Axis[L])
  peak + (logits - peak.broadcastTo(logits.shape)).exp.sum(Axis[L]).log

/** A one at the given position, which is where the record ends. */
private def isAt[Slot: Label, V: IsFloating](position: Tensor0[V], slots: AxisExtent[Slot], vtype: VType[V]): Tensor1[Slot, V] =
  val indices = Tensor1(slots.axis, VType[Int32]).fromArray(Array.range(0, slots.size)).asFloat(vtype)
  indices.elementEquals(position.broadcastTo(indices.shape)).asFloat(vtype)
