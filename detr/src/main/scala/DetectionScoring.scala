import dataset.ObjectClass
import deepwit.checkpointing.TensorTreeCheckpointer
import dimwit.*

import java.io.File

/** How far an object's defining points may be off, in pixels.
  *
  * A split is scored at every one of them, since a single threshold only says which side of
  * it the boxes fall on, not how far they still have to travel.
  */
val Tolerances = Seq(2f, 4f, 8f)

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

  /** One line of a score report: `<prefix>  <what>  <correct> / <total>  <percentage>`. */
  def report(prefix: String, what: String, correct: Int, total: Int): Unit =
    println(f"$prefix%5s  $what%-24s $correct%6d / $total%-6d ${100f * correct / total}%5.1f%%")

  /** `tolerance px`, the [[report]] prefix of a score that depends on the tolerance. */
  def at(tolerance: Float): String = f"$tolerance%2.0f px"

/** Reading the parameters back out of a training run's checkpoints. */
object Checkpoints:

  /** The newest checkpoint of the given run, or of the newest run under `root`.
    *
    * A checkpoint holds the whole training state, so this is also what a resumed run reads.
    */
  def loadLatest[S: TensorTree](root: String, run: Seq[String]): S =
    val directory = run.headOption.orElse(latestRun(root)).getOrElse(sys.error(s"no training run in $root"))
    val checkpointer = TensorTreeCheckpointer(directory)
    val step = checkpointer.iterations.lastOption.getOrElse(sys.error(s"no checkpoint in $directory"))
    println(s"loaded checkpoint $step of $directory")
    checkpointer.load[S](step).get

  /** The newest run directory under `root`, if it holds any. Runs are named after the time they
    * started, so the newest is the last by name.
    */
  def latestRun(root: String): Option[String] =
    Option(File(root).listFiles)
      .getOrElse(Array.empty[File])
      .filter(_.isDirectory)
      .map(_.getPath)
      .maxOption
