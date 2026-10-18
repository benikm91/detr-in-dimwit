package detr.floorplan.train

import detr.*
import detr.train.*
import detr.floorplan.model.*
import NodeHead.NodeLogits
import dimwit.*
import syntheticFloorplansDataset.FloorplanClass
import syntheticFloorplansDataset.FloorplanClasses
import syntheticFloorplansDataset.FloorplanNodes

import scala.language.implicitConversions

/** The set prediction loss of [[https://arxiv.org/abs/2005.12872 DETR]], over the nodes of a floor
  * plan rather than over boxes.
  *
  * Every node is answered by the query the optimal matching gives it, and every query left over
  * is trained towards [[FloorplanClass.NoNode]]. What a query costs against a node is the cross
  * entropy of the node's class and of the pixel of each coordinate of its two ends.
  */
class HungarianLoss[V: IsFloating](vtype: VType[V], canvas: Int) extends ((NodeLogits[V], FloorplanNodes[Node]) => Tensor0[V]):

  override def apply(logits: NodeLogits[V], target: FloorplanNodes[Node]): Tensor0[V] =
    val queries = logits.nodeClass.shape.extent(Axis[Query])
    val holdsNode = FloorplanClass.indicator(vtype)(_.isNode).slice(Axis[FloorplanClasses].at(target.nodeClass))
    val classCost = costOfValue(logits.nodeClass, target.nodeClass)
    val placementCost = placement(logits, target)

    // A slot holding no node costs every query the same, so only the nodes decide the matching.
    val answering = Matching.optimal((classCost + placementCost) *! holdsNode)
    val everyQuery = Tensor1(queries.axis, VType[Int32]).fromArray(Array.range(0, queries.size))
    val answers = (answering.broadcastTo(classCost.shape) elementEquals_! everyQuery).asFloat(vtype) *! holdsNode
    val answersNothing = 1f -! answers.sum(Axis[Node])

    val stops = costOfClass(logits.nodeClass, FloorplanClass.NoNode.id)
    val classification = ((classCost * answers).sum + (stops * answersNothing).sum) / queries.size.toFloat
    val placed = (placementCost * answers).sum / maximum(holdsNode.sum, 1f)
    classification + placed

  /** What every query's pixels would cost against where every node starts and ends. */
  private def placement(logits: NodeLogits[V], target: FloorplanNodes[Node]): Tensor2[Query, Node, V] =
    def placed(scores: Tensor2[Query, Pixel, V], coordinate: Tensor1[Node, Float32]) =
      costOfValue(scores, Pixels.of(coordinate, canvas))
    placed(logits.startX, target.startX) + placed(logits.startY, target.startY) + placed(logits.endX, target.endX) + placed(logits.endY, target.endY)
