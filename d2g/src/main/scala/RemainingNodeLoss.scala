import dataset.NodeClass
import dataset.NodeClasses
import dataset.Record
import dimwit.*
import dimwit.Conversions.given

import scala.language.implicitConversions

/** Equation 4 of the paper: the loss of remaining-node prediction over a graph record.
  *
  * A prediction embedding may answer with any node the slots before it have not taken, so its
  * cost is the smallest dissimilarity to any of them rather than the dissimilarity to one. Its
  * candidates stay on its side of the split a linearization keeps — the record nodes, then the
  * relationships — so a slot in the first block is never asked to name identifiers that do not
  * exist yet.
  *
  * Two terms complete it: the empty slot just past the record, so that transcription knows where
  * to stop, and the pass-through of every taken node, which is what keeps a node embedding
  * carrying its node while the prediction embedding beside it becomes another one.
  */
class RemainingNodeLoss[V: IsFloating](vtype: VType[V], canvas: Int) extends ((D2G.Scores[V], Record[Node]) => Tensor0[V]):

  /** Axis of the record's nodes seen as candidates to answer with rather than as positions. */
  private type Candidate = Prime[Node]

  override def apply(scored: D2G.Scores[V], target: Record[Node]): Tensor0[V] =
    val nodes = target.nodeClass.shape.extent(Axis[Node])
    val pairs = Shape2(nodes, Axis[Candidate] -> nodes.size)
    def classIs(holds: NodeClass => Boolean) =
      NodeClass.indicator(vtype)(holds).take(Axis[NodeClasses])(target.nodeClass)

    val holdsNode = classIs(_ != NodeClass.NoNode)
    val isRelationship = classIs(_.isRelationship)
    val numNodes = holdsNode.sum

    val sameBlock = 1f -! (isRelationship.broadcastTo(pairs) - isRelationship.relabelTo(Axis[Candidate]).broadcastTo(pairs)).abs
    val candidates = triu(Tensor(pairs, vtype).fill(1f)) * sameBlock * holdsNode.relabelTo(Axis[Candidate]).broadcastTo(pairs)

    val remaining = cheapest(dissimilarity(scored.remainingPredictions, target), candidates) * candidates.max(Axis[Candidate])
    val stops = costOfClass(scored.remainingPredictions.nodeClass, NodeClass.NoNode) * isAt(numNodes, nodes)
    val passedThrough = dissimilarity(scored.takenNodes, target) * Tensor2.eye(nodes, vtype) * holdsNode.broadcastTo(pairs)

    (remaining.sum + stops.sum) / (numNodes + Tensor0(vtype)(1f)) +
      passedThrough.sum / maximum(numNodes, Tensor0(vtype)(1f))

  /** What every position's prediction would cost against every node of the record: its class, and
    * what the *target* node carries — so that nothing depends on what the model predicts.
    */
  private def dissimilarity(logits: NodeLogits[V], target: Record[Node]): Tensor2[Node, Candidate, V] =
    val candidateClass = target.nodeClass.relabelTo(Axis[Candidate])
    def carried[Carries: Label](used: Tensor2[NodeClasses, Carries, V]) = used.take(Axis[NodeClasses])(candidateClass)
    val points = carried(NodeClass.usedPoints(vtype))
    costOfValue(logits.nodeClass, candidateClass) +
      carries(logits.xs, asCandidate(Pixels.of(target.xs, canvas)), points) +
      carries(logits.ys, asCandidate(Pixels.of(target.ys, canvas)), points) +
      carries(logits.links, asCandidate(target.links), carried(NodeClass.usedLinks(vtype)))

  private def carries[Carries: Label, Values: Label](
      logits: Tensor3[Node, Carries, Values, V],
      values: Tensor2[Candidate, Carries, Int32],
      used: Tensor2[Candidate, Carries, V]
  ): Tensor2[Node, Candidate, V] =
    val each = zipvmap(Axis[Carries])(logits, values):
      case (scores, wanted) => costOfValue(scores, wanted)
    (each * used.broadcastTo(each.shape)).sum(Axis[Carries])

  /** The smallest cost among the candidates of a position. The others are lifted above the whole
    * matrix rather than dropped, so that the minimum stays a plain reduction.
    */
  private def cheapest(cost: Tensor2[Node, Candidate, V], candidates: Tensor2[Node, Candidate, V]): Tensor1[Node, V] =
    val beyond = cost.max - cost.min + 1f
    where(candidates > Tensor.like(candidates).fill(0f), cost, cost +! beyond).min(Axis[Candidate])

  /** The cross entropy of every position's scores against the value every candidate holds. */
  private def costOfValue[L: Label](logits: Tensor2[Node, L, V], values: Tensor1[Candidate, Int32]): Tensor2[Node, Candidate, V] =
    val chosen = logits.take(Axis[L])(values)
    logNormalizer(logits).broadcastTo(chosen.shape) - chosen

  private def costOfClass(logits: Tensor2[Node, NodeClasses, V], nodeClass: NodeClass): Tensor1[Node, V] =
    logNormalizer(logits) - logits.slice(Axis[NodeClasses].at(nodeClass.id))

  private def logNormalizer[L: Label](logits: Tensor2[Node, L, V]): Tensor1[Node, V] =
    val peak = logits.max(Axis[L])
    peak + (logits - peak.broadcastTo(logits.shape)).exp.sum(Axis[L]).log

  /** A one at the given position, which is where the record ends. */
  private def isAt(position: Tensor0[V], nodes: AxisExtent[Node]): Tensor1[Node, V] =
    val indices = Tensor1(nodes.axis, VType[Int32]).fromArray(Array.range(0, nodes.size)).asFloat(vtype)
    indices.elementEquals(position.broadcastTo(indices.shape)).asFloat(vtype)

  private def asCandidate[Carries: Label](values: Tensor2[Node, Carries, Int32]): Tensor2[Candidate, Carries, Int32] =
    values.relabel(Axis[Node] -> Axis[Candidate])
