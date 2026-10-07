package d2s.train

import d2s.*
import d2s.model.*
import dataset.NodeClass
import dataset.NodeClasses
import dataset.RecordNodes
import NodeHead.NodeLogits
import dimwit.*

import scala.language.implicitConversions

/** What a record's nodes cost the model that writes them down.
  *
  * A prediction embedding may answer with any node the slots before it have not taken, so its cost
  * is the smallest dissimilarity to any of them rather than the dissimilarity to one. The slot just
  * past the last node is charged for saying so, which is how transcription knows where to stop.
  */
class RemainingNodeLoss[V: IsFloating](vtype: VType[V], canvas: Int)
    extends ((D2S.NodeQueryLogits[V], RecordNodes[Node]) => Tensor0[V]):

  /** Axis of the record's nodes seen as candidates to answer with rather than as positions. */
  private type Candidate = Prime[Node]

  override def apply(answered: D2S.NodeQueryLogits[V], target: RecordNodes[Node]): Tensor0[V] =
    require(answered.nodeClass.shape(Axis[PoolQuery]) == 2, "the loss pairs two answers")
    val (a, b) = (answered.at(0), answered.at(1))
    val nodes = target.nodeClass.shape.extent(Axis[Node])
    val pairs = Shape2(nodes, Axis[Candidate] -> nodes.size)
    val holdsNode = NodeClass.indicator(vtype)(_.isNode).slice(Axis[NodeClasses].at(target.nodeClass))
    val taken = holdsNode.sum
    val candidates = triu(Tensor(pairs, vtype).fill(1f)) *! holdsNode.relabelTo(Axis[Candidate])

    val asked = candidates.max(Axis[Candidate])
    val guessed = distinctly(dissimilarity(a, target), dissimilarity(b, target), candidates)
    val ended = isAt(taken, nodes, vtype)
    val stops = costOfClass(a.nodeClass, NodeClass.NoNode.id) + costOfClass(b.nodeClass, NodeClass.NoNode.id)
    // Both queries are charged, so the total is halved to stay on the scale of one answer.
    ((guessed * asked).sum + (stops * ended).sum) / ((taken + 1f) * 2f)

  /** What every position's scores would cost against every node of the record: its class, whether
    * it is construction geometry, and where the *target* node is placed — so that nothing depends
    * on what the model predicts. A class that runs nowhere is not measured on where it ends, nor
    * one that does not bend on its middle.
    */
  private def dissimilarity(logits: NodeLogits[V], target: RecordNodes[Node]): Tensor2[Node, Candidate, V] =
    val candidateClass = target.nodeClass.relabelTo(Axis[Candidate])
    def placed(scores: Tensor2[Node, Pixel, V], coordinate: Tensor1[Node, Float32]) =
      costOfValue(scores, Pixels.of(coordinate, canvas).relabelTo(Axis[Candidate]))
    val runsOn = NodeClass.indicator(vtype)(_.numPoints > 1).slice(Axis[NodeClasses].at(candidateClass))
    val bends = NodeClass.indicator(vtype)(_.numPoints > 2).slice(Axis[NodeClasses].at(candidateClass))
    val ends = placed(logits.endX, target.endX) + placed(logits.endY, target.endY)
    val middles = placed(logits.midX, target.midX) + placed(logits.midY, target.midY)
    costOfValue(logits.nodeClass, candidateClass) +
      costOfValue(logits.construction, target.construction.relabelTo(Axis[Candidate])) +
      placed(logits.startX, target.startX) +
      placed(logits.startY, target.startY) +
      ends *! runsOn +
      middles *! bends

/** What a slot's two queries cost when they must answer with two different remaining nodes.
  *
  * Neither query is privileged: what tells two of them apart is the node each settles on, not where
  * it ranks.
  */
def distinctly[Slot: Label, Candidate: Label, V: IsFloating](
    aScores: Tensor2[Slot, Candidate, V],
    bScores: Tensor2[Slot, Candidate, V],
    candidates: Tensor2[Slot, Candidate, V]
): Tensor1[Slot, V] =
  def onlyRemaining(scores: Tensor2[Slot, Candidate, V]) =
    val pastEveryCost = scores.max - scores.min + 1f
    where(candidates >! 0f, scores, scores +! pastEveryCost)

  def without(scores: Tensor2[Slot, Candidate, V], at: Tensor1[Slot, Int32]) =
    val slots = scores.shape.extent(Axis[Candidate])
    val indices = Tensor1(slots.axis, VType[Int32]).fromArray(Array.range(0, slots.size))
    val pastEveryCost = scores.max - scores.min + 1f
    where(at.broadcastTo(scores.shape) elementEquals_! indices, scores +! pastEveryCost, scores).min(Axis[Candidate])

  val (a, b) = (onlyRemaining(aScores), onlyRemaining(bScores))
  val (aAt, bAt) = (a.argmin(Axis[Candidate]), b.argmin(Axis[Candidate]))
  val (aMin, bMin) = (a.min(Axis[Candidate]), b.min(Axis[Candidate]))
  val (aNext, bNext) = (without(a, aAt), without(b, bAt))

  // Where they want the same node, whichever gives it up more cheaply is the one that moves. Two
  // different answers only exist while two nodes are left.
  val wantTheSame = aAt.elementEquals(bAt)
  val twoAreLeft = candidates.sum(Axis[Candidate]) >! 1.5f
  where(wantTheSame and twoAreLeft, minimum(aMin + bNext, bMin + aNext), aMin + bMin)

/** The cross entropy of every position's scores against the value every candidate holds. */
def costOfValue[Slot: Label, Candidate: Label, L: Label, V: IsFloating](
    logits: Tensor2[Slot, L, V],
    values: Tensor1[Candidate, Int32]
): Tensor2[Slot, Candidate, V] =
  val chosen = logits.slice(Axis[L].at(values))
  logNormalizer(logits) -! chosen

/** The cross entropy of every position's scores against one class, which is the one that ends the
  * record.
  */
def costOfClass[Slot: Label, Classes: Label, V: IsFloating](logits: Tensor2[Slot, Classes, V], id: Int): Tensor1[Slot, V] =
  logNormalizer(logits) - logits.slice(Axis[Classes].at(id))

def logNormalizer[Slot: Label, L: Label, V: IsFloating](logits: Tensor2[Slot, L, V]): Tensor1[Slot, V] =
  val peak = logits.max(Axis[L])
  peak + (logits -! peak).exp.sum(Axis[L]).log

/** A one at the given position, which is where the record ends. */
def isAt[Slot: Label, V: IsFloating](position: Tensor0[V], slots: AxisExtent[Slot], vtype: VType[V]): Tensor1[Slot, V] =
  val indices = Tensor1(slots.axis, VType[Int32]).fromArray(Array.range(0, slots.size)).asFloat(vtype)
  (indices elementEquals_! position).asFloat(vtype)
