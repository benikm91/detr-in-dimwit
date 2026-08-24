import dataset.ObjectClass
import dimwit.*

/** Scoring a detection against a target, in pixels rather than in the loss's terms.
  *
  * Shared by [[detrEval]] and anything scoring a model built on the detector, so that a node
  * of a graph is judged exactly as an object of a detection is.
  */
object DetectionScoring:

  /** One query slot in pixel coordinates. */
  case class Slot(objectClass: ObjectClass, centerX: Float, centerY: Float, width: Float, height: Float):
    def left: Float = centerX - width / 2
    def right: Float = centerX + width / 2
    def top: Float = centerY - height / 2
    def bottom: Float = centerY + height / 2
    def isHorizontal: Boolean = width >= height
    def isObject: Boolean = objectClass != ObjectClass.NoObject

  def slots(detection: ObjectDetection[Float32], imageWidth: Int, imageHeight: Int): Seq[Slot] =
    val label = detection.label.toArray
    val centerX = detection.box.centerX.toArray
    val centerY = detection.box.centerY.toArray
    val width = detection.box.width.toArray
    val height = detection.box.height.toArray
    label.indices.map: slot =>
      Slot(
        objectClass = ObjectClass.fromId(label(slot)),
        centerX = centerX(slot) * imageWidth,
        centerY = centerY(slot) * imageHeight,
        width = width(slot) * imageWidth,
        height = height(slot) * imageHeight
      )

  /** Whether a query got its slot right: the class, and the points that define the object to
    * within `tolerance` pixels — the two end points for a part line, the anchor for a text. A
    * slot holding no object is right when nothing was predicted in it.
    */
  def isDetected(target: Slot, predicted: Slot, tolerance: Float): Boolean =
    def near(expected: Float, actual: Float): Boolean = (expected - actual).abs <= tolerance
    target.objectClass == predicted.objectClass && (target.objectClass match
      case ObjectClass.NoObject => true
      case ObjectClass.Text     => near(target.centerX, predicted.centerX) && near(target.centerY, predicted.centerY)
      case ObjectClass.PartLine if target.isHorizontal =>
        near(target.left, predicted.left) && near(target.right, predicted.right) && near(target.centerY, predicted.centerY)
      case ObjectClass.PartLine =>
        near(target.top, predicted.top) && near(target.bottom, predicted.bottom) && near(target.centerX, predicted.centerX)
    )
