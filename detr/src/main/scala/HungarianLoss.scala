import dataset.Box
import dataset.Detection
import dataset.ObjectClass
import deepwit.base.softmax
import deepwit.loss.CategoricalCrossEntropy
import dimwit.*
import dimwit.Conversions.given

import scala.language.implicitConversions

/** The set prediction loss of the DETR paper.
  *
  * Predictions and targets are matched one to one on the same costs that are then optimized.
  * Target slots holding [[ObjectClass.NoObject]] are padding: they cost the same against every
  * prediction, so they only absorb the predictions no real target claims, which are then
  * trained towards "no object".
  */
class HungarianLoss[V: IsFloating](
    vtype: VType[V]
)(
    classWeight: Float = 1f,
    l1Weight: Float = 5f,
    giouWeight: Float = 2f,
    noObjectWeight: Float = 0.1f
) extends ((DETR.Prediction[V], ObjectDetection[V]) => Tensor0[V]):

  private type Target = Prime[BoundingBox]

  override def apply(prediction: DETR.Prediction[V], target: ObjectDetection[V]): Tensor0[V] =
    score(prediction, matchTargets(prediction, target))

  /** The targets in the order of the predictions they are matched to. */
  def matchTargets(prediction: DETR.Prediction[V], target: ObjectDetection[V]): ObjectDetection[V] =
    val padded = padToQueries(prediction, target)
    val matched = Matching.greedy(cost(prediction, padded))
    Detection(
      box = padded.box.map(_.take(Axis[BoundingBox])(matched)),
      label = padded.label.take(Axis[BoundingBox])(matched)
    )

  /** Pads the targets out to one slot per query.
    *
    * A dataset only carries as many slots as a sample can hold objects, while DETR predicts a
    * fixed set of queries and matches them one to one, so the surplus queries need a slot to be
    * matched against. Padding here, with [[ObjectClass.NoObject]], is what lets a prediction be
    * matched to "nothing" — and is why the query count stays a property of the model rather than
    * something the dataset has to be told.
    */
  private def padToQueries(prediction: DETR.Prediction[V], target: ObjectDetection[V]): ObjectDetection[V] =
    val queries = prediction.classLogits.shape(Axis[BoundingBox])
    val slots = target.label.shape(Axis[BoundingBox])
    require(
      queries >= slots,
      s"a model with $queries queries cannot cover $slots target slots: every query is matched to a distinct slot"
    )
    if queries == slots then target
    else
      val padding = Axis[BoundingBox] -> (queries - slots)
      // The boxes of padding slots are masked out of every term, so their value is arbitrary.
      val padBox = Tensor1(padding, vtype).fill(0f)
      val padLabel = Tensor1(padding, VType[Int32]).fill(ObjectClass.NoObject.id)
      Detection(
        box = target.box.map(concatenate(_, padBox, Axis[BoundingBox])),
        label = concatenate(target.label, padLabel, Axis[BoundingBox])
      )

  private def score(prediction: DETR.Prediction[V], target: ObjectDetection[V]): Tensor0[V] =
    val isObject = objectMask(target.label)
    val numObjects = maximum(isObject.sum, Tensor0(vtype)(1f))

    val classification = zipvmap(Axis[BoundingBox])(target.label, prediction.classLogits):
      case (objectClass, logits) => CategoricalCrossEntropy.fromLogits(objectClass, logits)
    val classificationLoss = (classification * (isObject *! (1f - noObjectWeight) +! noObjectWeight)).mean
    val boxLoss = (Box.l1(prediction.box, target.box) * isObject).sum
    val giouLoss = ((1f -! Box.giou(prediction.box, target.box)) * isObject).sum

    classificationLoss * classWeight + (boxLoss * l1Weight + giouLoss * giouWeight) / numObjects

  /** What every prediction would cost against every target.
    *
    * Padding targets carry a surcharge rather than a zero cost: a constant added to a column
    * leaves the optimal assignment alone, but it keeps a greedy matcher from handing its
    * cheapest predictions to slots that hold no object. The surcharge is derived from the
    * spread of the real costs so that it dominates them without swamping their precision.
    */
  private def cost(prediction: DETR.Prediction[V], target: ObjectDetection[V]): Tensor2[BoundingBox, Target, V] =
    val pairs = Shape2(
      prediction.classLogits.shape.extent(Axis[BoundingBox]),
      Axis[Target] -> target.label.shape(Axis[BoundingBox])
    )
    val targetClass = target.label.relabelTo(Axis[Target])
    val predicted = prediction.box.map(_.broadcastTo(pairs))
    val actual = target.box.map(_.relabelTo(Axis[Target]).broadcastTo(pairs))
    val probability = prediction.classLogits.vapply(Axis[ObjectClasses])(softmax)
    val classCost = -probability.take(Axis[ObjectClasses])(targetClass)
    val real = classCost *! classWeight +
      Box.l1(predicted, actual) *! l1Weight +
      (1f -! Box.giou(predicted, actual)) *! giouWeight
    val padding = (1f -! objectMask(targetClass)).broadcastTo(pairs)
    real + padding *! (real.max - real.min + 1f)

  private def objectMask[L: Label](classes: Tensor1[L, Int32]): Tensor1[L, V] =
    (classes > Tensor.like(classes).fill(ObjectClass.NoObject.id)).asFloat(vtype)
