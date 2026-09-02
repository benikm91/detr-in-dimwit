package egtr

import dataset.ObjectClass
import dataset.RelationClasses
import deepwit.activation.sigmoid
import detr.*
import dimwit.*
import dimwit.Conversions.given

import scala.language.implicitConversions

/** The multi-task loss of [[https://arxiv.org/abs/2404.02072 EGTR]], equation 7: the set
  * prediction loss of the detector, the relations, and the connectivity of a pair as an
  * auxiliary task.
  *
  * The detection is matched exactly once, by [[HungarianLoss]], and everything else is read
  * through that matching: the query a target object was assigned answers for it, so the edge
  * between two queries is the edge between the two slots they answer for. Queries assigned to
  * padding answer for no object and read a padding row, which carries no edges.
  *
  * The graph is far more empty than not, and the three regions of it are not equally
  * informative (§3.3.1, fig. 4): a pair of detected objects that carries no edge is a genuine
  * negative, while a pair reaching into the padding says only that a query found nothing. They
  * are weighted apart rather than sampled apart — see `README.md`.
  *
  * Relation labels are smoothed by how well the two objects were detected (§3.3.1, eq. 8), so
  * that a relation between two objects the detector has not found yet is not insisted on. Early
  * in training that leaves the detection loss to lead and the relation loss to follow, which is
  * the curriculum the paper is after.
  *
  * @param detection          The set prediction loss of the detector, and the matcher.
  * @param relationWeight     `λ_rel` of eq. 7.
  * @param connectivityWeight `λ_con` of eq. 7.
  * @param minimumUncertainty `α` of eq. 8: the uncertainty of a query that hit its object
  *                           exactly. It also sets the scale of the smoothing — a query costing
  *                           `-logit(α)` more than a perfect match is half uncertain, and its
  *                           relations count half.
  * @param negativeWeight     What a pair of detected objects without an edge weighs against a
  *                           pair with one.
  * @param nonMatchingWeight  What a pair involving a query matched to no object weighs. These
  *                           make up almost the whole graph, and carry the least.
  */
class EGTRLoss[V: IsFloating](vtype: VType[V], val detection: HungarianLoss[V])(
    relationWeight: Float = 15f,
    connectivityWeight: Float = 30f,
    minimumUncertainty: Float = 0.02f,
    negativeWeight: Float = 1f,
    nonMatchingWeight: Float = 0.1f
) extends ((EGTR.Prediction[V], SceneGraph[V]) => Tensor0[V]):

  import EGTRLoss.Cost
  import EGTRLoss.Target

  override def apply(prediction: EGTR.Prediction[V], target: SceneGraph[V]): Tensor0[V] =
    total(cost(prediction, target))

  /** Eq. 7: the three terms weighted into the one number that is optimized. */
  def total(cost: Cost[V]): Tensor0[V] =
    cost.detection + cost.relation * relationWeight + cost.connectivity * connectivityWeight

  /** The three terms of eq. 7, before they are weighted together. */
  def cost(prediction: EGTR.Prediction[V], target: SceneGraph[V]): Cost[V] =
    val matched = detection.matched(prediction.detection, target.objects)
    val edges = graph(matched, target.relations)
    Cost(
      detection = detection.score(prediction.detection, matched.targets),
      relation = weightedCrossEntropy(edges.smoothed, prediction.graph.relationLogits, edges.weight),
      // A pair is connected when it carries any relation at all, so both its target and the
      // region it falls in are the strongest of those of its relations.
      connectivity = weightedCrossEntropy(
        edges.smoothed.max(Axis[RelationClasses]),
        prediction.graph.connectivityLogits,
        edges.weight.max(Axis[RelationClasses])
      )
    )

  /** The target graph over the query slots: query `i` answers for target slot `slot(i)`, so the
    * edge between two queries is the edge between the two slots they answer for. A query matched
    * to padding reads a padding row, which carries no edges.
    *
    * This is the permutation of §3.3.1, and what a prediction is both trained and scored against.
    */
  def targetGraph(matched: HungarianLoss.Match[V], relations: Tensor3[BoundingBox, RelatedBox, RelationClasses, V]): Tensor3[BoundingBox, RelatedBox, RelationClasses, V] =
    padToQueries(relations, matched.cost.shape.extent(Axis[BoundingBox]))
      .take(Axis[BoundingBox])(matched.slot)
      .take(Axis[RelatedBox])(matched.slot.relabelTo(Axis[RelatedBox]))

  /** The target graph, lined up with the queries by the matching and smoothed by how well each
    * of them did, together with the weight every entry carries.
    */
  private def graph(matched: HungarianLoss.Match[V], relations: Tensor3[BoundingBox, RelatedBox, RelationClasses, V]): Target[V] =
    val queries = matched.cost.shape.extent(Axis[BoundingBox])
    val grid = Shape3(
      queries,
      Axis[RelatedBox] -> queries.size,
      relations.shape.extent(Axis[RelationClasses])
    )

    val edges = targetGraph(matched, relations)
    val isObject = objectMask(matched.targets.label)
    val isObjectPair = isObject.broadcastTo(grid) * isObject.relabelTo(Axis[RelatedBox]).broadcastTo(grid)
    val certainty = 1f -! uncertainty(matched.cost)

    Target(
      smoothed = edges * certainty.broadcastTo(grid) * certainty.relabelTo(Axis[RelatedBox]).broadcastTo(grid),
      // The three regions, as one weight per entry: an edge of the target graph weighs 1, a pair
      // of target objects without an edge between them `negativeWeight`, and everything else — a
      // pair at least one end of which answers for no object — `nonMatchingWeight`.
      weight = edges *! (1f - negativeWeight) + isObjectPair *! (negativeWeight - nonMatchingWeight) +! nonMatchingWeight
    )

  /** Eq. 8: how uncertain the object a query answers for is, from what its match cost against
    * the cost of hitting the target exactly.
    *
    * No gradient flows back through the cost. It is a measurement of the detection that shapes a
    * label, and a label is a constant — left differentiable it becomes a way to make the relation
    * loss cheap by detecting badly, since a query uncertain enough has all of its relations
    * damped to zero. The matcher this cost comes from runs without gradients in the paper's
    * implementation for the same reason.
    */
  private def uncertainty(cost: Tensor1[BoundingBox, V]): Tensor1[BoundingBox, V] =
    sigmoid(detached(cost) -! (detection.minimumCost - logit(minimumUncertainty)))

  /** The same values, with the gradient stopped at them. dimwit has no such operation, so this
    * reaches for `jax.lax.stop_gradient` directly.
    */
  private def detached[T <: Tuple: Labels](tensor: Tensor[T, V]): Tensor[T, V] =
    dimwit.python.PyBridge.liftPyTensor[T, V](
      dimwit.jax.Jax.lax.stop_gradient(dimwit.python.PyBridge.toPyTensor(tensor))
    )

  private def logit(probability: Float): Float =
    Math.log(probability / (1f - probability)).toFloat

  /** Pads the target graph out to one row and column per query, as [[HungarianLoss]] pads the
    * target objects, so that the matching's slot indices address it.
    */
  private def padToQueries(
      relations: Tensor3[BoundingBox, RelatedBox, RelationClasses, V],
      queries: AxisExtent[BoundingBox]
  ): Tensor3[BoundingBox, RelatedBox, RelationClasses, V] =
    val slots = relations.shape(Axis[BoundingBox])
    if slots == queries.size then relations
    else
      val predicates = relations.shape.extent(Axis[RelationClasses])
      val padding = Axis[BoundingBox] -> (queries.size - slots)
      val rows = Tensor(Shape3(padding, Axis[RelatedBox] -> slots, predicates), vtype).fill(0f)
      val columns = Tensor(Shape3(queries, Axis[RelatedBox] -> (queries.size - slots), predicates), vtype).fill(0f)
      concatenate(concatenate(relations, rows, Axis[BoundingBox]), columns, Axis[RelatedBox])

  private def weightedCrossEntropy[T <: Tuple: Labels](target: Tensor[T, V], logits: Tensor[T, V], weight: Tensor[T, V]): Tensor0[V] =
    (crossEntropy(target, logits) * weight).sum / weight.sum

  /** Binary cross entropy from logits, entry by entry, by the stable identity
    * [[deepwit.loss.BinaryCrossEntropy.fromLogits]] uses on a scalar. Sigmoid rather than
    * softmax, since a pair may carry any number of relations.
    */
  private def crossEntropy[T <: Tuple: Labels](target: Tensor[T, V], logits: Tensor[T, V]): Tensor[T, V] =
    maximum(logits, Tensor.like(logits).fill(0f)) - logits * target + (1f +! (-logits.abs).exp).log

  /** Which slots hold an object rather than padding, as ones and zeros. */
  private def objectMask(classes: Tensor1[BoundingBox, Int32]): Tensor1[BoundingBox, V] =
    (classes > Tensor.like(classes).fill(ObjectClass.NoObject.id)).asFloat(vtype)

object EGTRLoss:

  /** The terms of eq. 7 before they are weighted together, so that they can be reported apart. */
  case class Cost[V](
      detection: Tensor0[V],
      relation: Tensor0[V],
      connectivity: Tensor0[V]
  )

  /** The target graph over the query slots.
    *
    * @param smoothed The target of every entry, an edge damped by the uncertainty of the two
    *                 objects it relates.
    * @param weight   How much every entry counts, by the region of the graph it falls in.
    */
  private case class Target[V](
      smoothed: Tensor3[BoundingBox, RelatedBox, RelationClasses, V],
      weight: Tensor3[BoundingBox, RelatedBox, RelationClasses, V]
  )
