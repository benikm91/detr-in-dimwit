import dataset.Detection
import dataset.ObjectClass
import deepwit.base.softmax
import deepwit.loss.CategoricalCrossEntropy
import dimwit.*
import dimwit.Conversions.given

import scala.language.implicitConversions

/** The set prediction loss of the DETR paper.
  *
  * Predictions and targets are matched one to one by [[HungarianMatching]] on the same
  * costs that are then optimized. Target slots holding [[ObjectClass.NoObject]] are padding:
  * they cost the same against every prediction, so they only absorb the predictions no real
  * target claims, which are then trained towards "no object".
  */
case class HungarianLoss[V: IsFloating](
    classWeight: Float = 1f,
    l1Weight: Float = 5f,
    giouWeight: Float = 2f,
    noObjectWeight: Float = 0.1f
) extends ((DETR.Prediction[V], ObjectDetection) => Tensor0[V]):

  private type Target = Prime[BoundingBox]

  /** The targets in the order of the predictions they are matched to.
    *
    * The assignment is a discrete decision on concrete values, so this cannot be traced by
    * `grad` or `jit` — match first, then differentiate [[apply]] against the result.
    */
  def matchTargets(prediction: DETR.Prediction[V], target: ObjectDetection): ObjectDetection =
    val matched = HungarianMatching(cost(prediction, target))
    Detection(
      centerX = target.centerX.take(Axis[BoundingBox])(matched),
      centerY = target.centerY.take(Axis[BoundingBox])(matched),
      width = target.width.take(Axis[BoundingBox])(matched),
      height = target.height.take(Axis[BoundingBox])(matched),
      label = target.label.take(Axis[BoundingBox])(matched)
    )

  /** Scores every prediction against the target it was matched with by [[matchTargets]]. */
  override def apply(prediction: DETR.Prediction[V], matchedTarget: ObjectDetection): Tensor0[V] =
    val predicted = predictedBox(prediction)
    val target = targetBox(matchedTarget)
    val isObject = objectMask(matchedTarget.label)
    val numObjects = maximum(isObject.sum, Tensor0(VType[V])(1f))

    val classification = zipvmap(Axis[BoundingBox])(matchedTarget.label, prediction.classLogits):
      case (objectClass, logits) => CategoricalCrossEntropy.fromLogits(objectClass, logits)
    val classificationLoss = (classification * (isObject *! (1f - noObjectWeight) +! noObjectWeight)).mean
    val boxLoss = (Box.l1(predicted, target) * isObject).sum
    val giouLoss = ((1f -! Box.giou(predicted, target)) * isObject).sum

    classificationLoss * classWeight + (boxLoss * l1Weight + giouLoss * giouWeight) / numObjects

  private def cost(prediction: DETR.Prediction[V], target: ObjectDetection): Tensor2[BoundingBox, Target, V] =
    val pairs = Shape2(
      Axis[BoundingBox] -> prediction.centerX.shape(Axis[BoundingBox]),
      Axis[Target] -> target.label.shape(Axis[BoundingBox])
    )
    val targetClass = target.label.relabelTo(Axis[Target])
    val predicted = predictedBox(prediction).map(_.broadcastTo(pairs))
    val actual = targetBox(target).map(_.relabelTo(Axis[Target]).broadcastTo(pairs))
    val probability = prediction.classLogits.vapply(Axis[ObjectClasses])(softmax)
    val classCost = -probability.take(Axis[ObjectClasses])(targetClass).transpose
    val boxCost = Box.l1(predicted, actual) *! l1Weight
    val giouCost = (1f -! Box.giou(predicted, actual)) *! giouWeight
    (classCost *! classWeight + boxCost + giouCost) * objectMask(targetClass).broadcastTo(pairs)

  private def predictedBox(prediction: DETR.Prediction[V]): Box[Tuple1[BoundingBox], V] =
    Box(prediction.centerX, prediction.centerY, prediction.width, prediction.height)

  private def targetBox(target: ObjectDetection): Box[Tuple1[BoundingBox], V] =
    Box(
      centerX = target.centerX.asFloat(VType[V]),
      centerY = target.centerY.asFloat(VType[V]),
      width = target.width.asFloat(VType[V]),
      height = target.height.asFloat(VType[V])
    )

  private def objectMask[L: Label](classes: Tensor1[L, Int32]): Tensor1[L, V] =
    (classes > Tensor.like(classes).fill(ObjectClass.NoObject.id)).asFloat(VType[V])
