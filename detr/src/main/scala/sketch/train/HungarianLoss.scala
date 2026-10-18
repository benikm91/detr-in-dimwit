package detr.sketch.train

import detr.*
import detr.train.*
import detr.sketch.model.*
import dataset.NodeClass
import dataset.NodeClasses
import dataset.RecordNodes
import NodeHead.NodeLogits
import dimwit.*

import scala.language.implicitConversions

/** The set prediction loss of [[https://arxiv.org/abs/2005.12872 DETR]], over the nodes of a
  * record rather than over boxes.
  *
  * Every node is answered by the query the optimal matching gives it, and every query left over
  * is trained towards [[NodeClass.NoNode]]. What a query costs against a node is what the
  * transcriber's node head pays for it: the cross entropy of the node's class, of whether it is
  * construction geometry, and of the pixel of every point that class places.
  */
class HungarianLoss[V: IsFloating](vtype: VType[V], canvas: Int) extends ((NodeLogits[V], RecordNodes[Node]) => Tensor0[V]):

  override def apply(logits: NodeLogits[V], target: RecordNodes[Node]): Tensor0[V] =
    val queries = logits.nodeClass.shape.extent(Axis[Query])
    val holdsNode = NodeClass.indicator(vtype)(_.isNode).slice(Axis[NodeClasses].at(target.nodeClass))
    val classCost = costOfValue(logits.nodeClass, target.nodeClass)
    val placementCost = costOfValue(logits.construction, target.construction) + placement(logits, target)

    // A slot holding no node costs every query the same, so only the nodes decide the matching.
    val answering = Matching.optimal((classCost + placementCost) *! holdsNode)
    val answers = (answering.broadcastTo(classCost.shape) elementEquals_! indices(queries)).asFloat(vtype) *! holdsNode
    val answersNothing = 1f -! answers.sum(Axis[Node])

    val stops = costOfClass(logits.nodeClass, NodeClass.NoNode.id)
    val classification = ((classCost * answers).sum + (stops * answersNothing).sum) / queries.size.toFloat
    val placed = (placementCost * answers).sum / maximum(holdsNode.sum, 1f)
    classification + placed

  /** What every query's pixels would cost against where every node is placed. A class that runs
    * nowhere is not measured on where it ends, nor one that does not bend on its middle.
    */
  private def placement(logits: NodeLogits[V], target: RecordNodes[Node]): Tensor2[Query, Node, V] =
    def placed(scores: Tensor2[Query, Pixel, V], coordinate: Tensor1[Node, Float32]) =
      costOfValue(scores, Pixels.of(coordinate, canvas))
    val runsOn = NodeClass.indicator(vtype)(_.numPoints > 1).slice(Axis[NodeClasses].at(target.nodeClass))
    val bends = NodeClass.indicator(vtype)(_.numPoints > 2).slice(Axis[NodeClasses].at(target.nodeClass))
    val ends = placed(logits.endX, target.endX) + placed(logits.endY, target.endY)
    val middles = placed(logits.midX, target.midX) + placed(logits.midY, target.midY)
    placed(logits.startX, target.startX) + placed(logits.startY, target.startY) + ends *! runsOn + middles *! bends

  private def indices[L: Label](extent: AxisExtent[L]): Tensor1[L, Int32] =
    Tensor1(extent.axis, VType[Int32]).fromArray(Array.range(0, extent.size))
