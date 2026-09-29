package detr.train

import detr.*
import detr.model.*
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
  * transcriber's node head pays for it: the cross entropy of the node's class, and of the pixel of
  * every point that class places.
  */
class HungarianLoss[V: IsFloating](vtype: VType[V], canvas: Int) extends ((NodeLogits[V], RecordNodes[Node]) => Tensor0[V]):

  override def apply(logits: NodeLogits[V], target: RecordNodes[Node]): Tensor0[V] =
    val queries = logits.nodeClass.shape.extent(Axis[Query])
    val holdsNode = NodeClass.indicator(vtype)(_.isNode).take(Axis[NodeClasses])(target.nodeClass)
    val classCost = costOfValue(logits.nodeClass, target.nodeClass)
    val placementCost = placement(logits, target)

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
    val runsOn = NodeClass.indicator(vtype)(_.numPoints > 1).take(Axis[NodeClasses])(target.nodeClass)
    val bends = NodeClass.indicator(vtype)(_.numPoints > 2).take(Axis[NodeClasses])(target.nodeClass)
    val ends = placed(logits.endX, target.endX) + placed(logits.endY, target.endY)
    val middles = placed(logits.midX, target.midX) + placed(logits.midY, target.midY)
    placed(logits.startX, target.startX) + placed(logits.startY, target.startY) + ends *! runsOn + middles *! bends

  /** The cross entropy of every query's scores against the value every node holds. */
  private def costOfValue[L: Label](logits: Tensor2[Query, L, V], values: Tensor1[Node, Int32]): Tensor2[Query, Node, V] =
    logNormalizer(logits) -! logits.take(Axis[L])(values)

  /** The cross entropy of every query's scores against one class. */
  private def costOfClass(logits: Tensor2[Query, NodeClasses, V], id: Int): Tensor1[Query, V] =
    logNormalizer(logits) - logits.slice(Axis[NodeClasses].at(id))

  private def logNormalizer[L: Label](logits: Tensor2[Query, L, V]): Tensor1[Query, V] =
    val peak = logits.max(Axis[L])
    peak + (logits -! peak).exp.sum(Axis[L]).log

  private def indices[L: Label](extent: AxisExtent[L]): Tensor1[L, Int32] =
    Tensor1(extent.axis, VType[Int32]).fromArray(Array.range(0, extent.size))
