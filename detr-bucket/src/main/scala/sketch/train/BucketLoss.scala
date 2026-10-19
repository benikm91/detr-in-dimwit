package detrbucket.sketch.train

import detr.*
import detr.train.*
import detrbucket.sketch.Buckets
import detrbucket.sketch.model.*
import BucketHead.NodeLogits
import dataset.NodeClass
import dataset.NodeClasses
import dataset.RecordNodes
import dimwit.*

import scala.language.implicitConversions

/** The set prediction loss of [[https://arxiv.org/abs/2005.12872 DETR]], taken bucket by bucket.
  *
  * Every node is answered by the query the optimal matching of its class's bucket gives it, and
  * every query left over is trained towards answering with none. What a query costs against a node
  * is what [[detr.sketch.train.HungarianLoss]] charges, with its bucket's class in place of a class
  * of its own: the cross entropy of answering, of whether the node is construction geometry, and of
  * the pixel of every point the class places.
  */
class BucketLoss[V: IsFloating](vtype: VType[V], canvas: Int, buckets: Buckets) extends ((NodeLogits[V], RecordNodes[Node]) => Tensor0[V]):

  override def apply(logits: NodeLogits[V], target: RecordNodes[Node]): Tensor0[V] =
    val queries = logits.presence.shape.extent(Axis[Query])
    val holdsNode = NodeClass.indicator(vtype)(_.isNode).slice(Axis[NodeClasses].at(target.nodeClass))
    val answering = costOfClass(logits.presence, 1)
    val placementCost = costOfValue(logits.construction, target.construction) + placement(logits, target)

    // Within a bucket, only the nodes of its class decide the matching, and only they are answered.
    val answers = buckets.ranges
      .map: (nodeClass, range) =>
        val ofClass = (target.nodeClass elementEquals_! nodeClass.id).asFloat(vtype)
        val inBucket = Axis[Query].at(range)
        val cost = (placementCost.slice(inBucket) +! answering.slice(inBucket)) *! ofClass
        val answeredBy = Matching.optimal(cost)
        val bucketQueries = Tensor1(Axis[Query], VType[Int32]).fromArray(Array.range(0, range.size))
        (answeredBy.broadcastTo(cost.shape) elementEquals_! bucketQueries).asFloat(vtype) *! ofClass
      .reduce((before, after) => concatenate(before, after, Axis[Query]))
    val answersNode = answers.sum(Axis[Node])

    val stops = costOfClass(logits.presence, 0)
    val presence = ((answering * answersNode).sum + (stops * (1f -! answersNode)).sum) / queries.size.toFloat
    val placed = (placementCost * answers).sum / maximum(holdsNode.sum, 1f)
    presence + placed

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
