package d2seq.train

import d2seq.*
import d2seq.model.NodeHead.NodeLogits
import dataset.NodeClass
import dataset.NodeClasses
import dataset.RecordNodes
import dimwit.*

import scala.language.implicitConversions

/** What a record's nodes cost the model that writes them one after the other: at every slot, the
  * node the order puts there, up to the slot just past the last node, which is charged for saying
  * the record ends there. A class that runs nowhere is not measured on where it ends, nor one that
  * does not bend on its middle.
  */
class NextNodeLoss[V: IsFloating](vtype: VType[V], canvas: Int) extends ((NodeLogits[V], RecordNodes[Node]) => Tensor0[V]):

  override def apply(answered: NodeLogits[V], target: RecordNodes[Node]): Tensor0[V] =
    def holds(has: NodeClass => Boolean) = NodeClass.indicator(vtype)(has).slice(Axis[NodeClasses].at(target.nodeClass))
    def placed(scores: Tensor2[Node, Pixel, V], coordinate: Tensor1[Node, Float32]) = crossEntropy(scores, Pixels.of(coordinate, canvas))
    val taken = holds(_.isNode).sum
    val slots = target.nodeClass.shape.extent(Axis[Node])
    val charged = (Tensor1(slots.axis).fromRange(0 until slots.size).asFloat(vtype) <=! taken).asFloat(vtype)
    val cost = crossEntropy(answered.nodeClass, target.nodeClass) +
      (crossEntropy(answered.construction, target.construction) + placed(answered.startX, target.startX) + placed(answered.startY, target.startY)) * holds(_.isNode) +
      (placed(answered.endX, target.endX) + placed(answered.endY, target.endY)) * holds(_.numPoints > 1) +
      (placed(answered.midX, target.midX) + placed(answered.midY, target.midY)) * holds(_.numPoints > 2)
    (cost * charged).sum / (taken + 1f)

/** The cross entropy of every slot's scores against the value at that slot. */
private def crossEntropy[L: Label, V: IsFloating](logits: Tensor2[Node, L, V], values: Tensor1[Node, Int32]): Tensor1[Node, V] =
  zipvmap(Axis[Node])(logits, values): (scores, value) =>
    -scores.logSoftmax(Axis[L]).slice(Axis[L].at(value))
