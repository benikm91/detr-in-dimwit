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
class RemainingNodeLoss[V: IsFloating](vtype: VType[V], canvas: Int, withPassThrough: Boolean = true, separation: Float = 1f):

  /** Axis of the record's nodes seen as candidates to answer with rather than as positions. */
  private type Candidate = Prime[Node]

  def apply(scored: D2G.NodeScores[V], target: RecordNodes[Node]): Answered[V] =
    val nodes = target.nodeClass.shape.extent(Axis[Node])
    val pairs = Shape2(nodes, Axis[Candidate] -> nodes.size)
    val holdsNode = NodeClass.indicator(vtype)(_.isDrawn).take(Axis[NodeClasses])(target.nodeClass)
    val taken = holdsNode.sum
    val candidates = triu(Tensor(pairs, vtype).fill(1f)) * holdsNode.relabelTo(Axis[Candidate]).broadcastTo(pairs)

    val asked = candidates.max(Axis[Candidate])
    // Each token answers with whatever remaining node it likes and is charged for that one alone,
    // so its target is its own choice and never a node handed to it because the other got there
    // first. What keeps the two apart is a separate cost on agreeing, which they can always choose
    // to pay — and at the end of a record, where two different answers may be impossible, paying
    // it is what they should do.
    val chosen = scored.remaining.toSeq.map(logits => cheapest(dissimilarity(logits, target), candidates)).reduce(_ + _)
    val agreed = agreement(scored.remaining.one, scored.remaining.other) * enoughLeft(candidates, vtype)
    val ended = isAt(taken, nodes, vtype)
    val stops = scored.remaining.toSeq.map(logits => costOfClass(logits.nodeClass, NodeClass.NoNode.id)).reduce(_ + _)
    // Both tokens are charged, so the cost is halved to stay on the scale of one answer.
    val together = (agreed * asked).sum * separation
    val answered = ((chosen * asked).sum + (stops * ended).sum + together) / ((taken + 1f) * PredictionsPerSlot.toFloat)

    val cost =
      if !withPassThrough then answered
      else
        val passedThrough = dissimilarity(scored.taken, target) * Tensor2.eye(nodes, vtype) * holdsNode.broadcastTo(pairs)
        answered + passedThrough.sum / maximum(taken, 1f)
    Answered(cost, (agreed * asked).sum, asked.sum)

  /** How likely the two tokens of a position name the same node: the chance that a draw from each
    * agrees on the class and on where the node starts, which is what tells two of them apart.
    */
  private def agreement(one: NodeLogits[V], other: NodeLogits[V]): Tensor1[Node, V] =
    overlap(one.nodeClass, other.nodeClass) * overlap(one.startX, other.startX) * overlap(one.startY, other.startY)

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
class RemainingEdgeLoss[V: IsFloating](vtype: VType[V], withPassThrough: Boolean = true, separation: Float = 1f):

  /** Axis of the record's relationships seen as candidates to answer with rather than as
    * positions.
    */
  private type Candidate = Prime[Edge]

  def apply(scored: D2G.EdgeScores[V], target: RecordEdges[Edge]): Answered[V] =
    val edges = target.edgeClass.shape.extent(Axis[Edge])
    val pairs = Shape2(edges, Axis[Candidate] -> edges.size)
    val holdsEdge = EdgeClass.indicator(vtype)(_.relates).take(Axis[EdgeClasses])(target.edgeClass)
    val taken = holdsEdge.sum
    val candidates = triu(Tensor(pairs, vtype).fill(1f)) * holdsEdge.relabelTo(Axis[Candidate]).broadcastTo(pairs)

    val asked = candidates.max(Axis[Candidate])
    val chosen = scored.remaining.toSeq.map(logits => cheapest(dissimilarity(logits, target), candidates)).reduce(_ + _)
    val agreed = agreement(scored.remaining.one, scored.remaining.other) * enoughLeft(candidates, vtype)
    val ended = isAt(taken, edges, vtype)
    val stops = scored.remaining.toSeq.map(logits => costOfClass(logits.edgeClass, EdgeClass.NoEdge.id)).reduce(_ + _)
    val together = (agreed * asked).sum * separation
    val answered = ((chosen * asked).sum + (stops * ended).sum + together) / ((taken + 1f) * PredictionsPerSlot.toFloat)

    val cost =
      if !withPassThrough then answered
      else
        val passedThrough = dissimilarity(scored.taken, target) * Tensor2.eye(edges, vtype) * holdsEdge.broadcastTo(pairs)
        answered + passedThrough.sum / maximum(taken, 1f)
    Answered(cost, (agreed * asked).sum, asked.sum)

  /** The same for a relationship: two of them are told apart by what they are and by the nodes they
    * name.
    */
  private def agreement(one: EdgeLogits[V], other: EdgeLogits[V]): Tensor1[Edge, V] =
    overlap(one.edgeClass, other.edgeClass) * overlap(one.subject, other.subject) * overlap(one.obj, other.obj)

  /** The same for a relationship, which carries the two nodes it relates where a node carries the
    * points it is placed by. Only relationships are ever candidates, so both ends always count.
    */
  private def dissimilarity(logits: EdgeLogits[V], target: RecordEdges[Edge]): Tensor2[Edge, Candidate, V] =
    def named(scores: Tensor2[Edge, LinkedNode, V], end: Tensor1[Edge, Int32]) =
      costOfValue(scores, end.relabelTo(Axis[Candidate]))
    costOfValue(logits.edgeClass, target.edgeClass.relabelTo(Axis[Candidate])) +
      named(logits.subject, target.subject) +
      named(logits.obj, target.obj)

/** What a record's remaining prediction cost, and how much its tokens agreed.
  *
  * `collided` counts the slots whose two tokens answered with the same node, out of `asked`. It
  * measures nothing about accuracy — it says whether the noise is doing its job at all. Near
  * `asked` and the two tokens are the same token and the whole exercise is inert; near zero and
  * they are answering independently, which is what the loss is asking of them.
  */
case class Answered[V](cost: Tensor0[V], collided: Tensor0[V], asked: Tensor0[V])

/** Whether a position has two remaining nodes to tell apart.
  *
  * On the last one there is only one answer to give, so both tokens must give it and neither is
  * charged for agreeing — the alternative they are being asked for does not exist.
  */
private def enoughLeft[Slot: Label, Candidate: Label, V: IsFloating](
    candidates: Tensor2[Slot, Candidate, V],
    vtype: VType[V]
): Tensor1[Slot, V] =
  val left = candidates.sum(Axis[Candidate])
  (left > Tensor.like(left).fill(1.5f)).asFloat(vtype)

/** The chance that a draw from each of two scored positions lands on the same value. */
private def overlap[Slot: Label, L: Label, V: IsFloating](
    one: Tensor2[Slot, L, V],
    other: Tensor2[Slot, L, V]
): Tensor1[Slot, V] =
  (likelihood(one) * likelihood(other)).sum(Axis[L])

private def likelihood[Slot: Label, L: Label, V: IsFloating](logits: Tensor2[Slot, L, V]): Tensor2[Slot, L, V] =
  (logits - logNormalizer(logits).broadcastTo(logits.shape)).exp

/** The smallest cost among the candidates of a position. The others are lifted above the whole
  * matrix rather than dropped, so that the minimum stays a plain reduction.
  */
private def cheapest[Slot: Label, Candidate: Label, V: IsFloating](
    cost: Tensor2[Slot, Candidate, V],
    candidates: Tensor2[Slot, Candidate, V]
): Tensor1[Slot, V] =
  val beyond = cost.max - cost.min + 1f
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
