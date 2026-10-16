package d2s.train

import dimwit.*

import scala.language.implicitConversions

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
