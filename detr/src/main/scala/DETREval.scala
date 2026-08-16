import dataset.Detection
import dataset.LShapeDetectionDataset
import dataset.LShapeDetectionDataset.Split
import dataset.ObjectClass
import dataset.Outlines
import deepwit.checkpointing.TensorTreeCheckpointer
import dimwit.*
import plotwit.*
import viz.PlotTargets.websocket

import java.io.File

/** How far an object's defining points may be off, in pixels.
  *
  * The split is scored at every one of them, since a single threshold only says which side of
  * it the boxes fall on, not how far they still have to travel.
  */
private val Tolerances = Seq(2f, 4f, 8f)

/** Plots what a trained model detects: `sbt "detr/runMain detrPlot"`.
  *
  * Shows the first drawings of the validation and the training split on their own, with
  * their targets, and with what the model predicts. Note that touching the training split
  * downloads 8.6 GB on first use.
  */
@main
def detrPlot(run: String*): Unit =
  dimwit.initialize()

  val model = load(run)
  val rows = Seq(Split.Validation, Split.Train).flatMap: split =>
    val data = LShapeDetectionDataset.open(Axis[Width], Axis[Height], Axis[Channel], Axis[BoundingBox])(split)
    data
      .samples()
      .take(3)
      .zipWithIndex
      .map: (sample, index) =>
        val drawing = Outlines.greyLevels(sample.image)
        Seq(
          plots.imagePlot(drawing, _.title := s"${split.fileName} $index"),
          plots.imagePlot(Outlines(drawing, sample.objects), _.title := s"${split.fileName} $index — target"),
          plots.imagePlot(Outlines(drawing, model(sample.image)), _.title := s"${split.fileName} $index — predicted")
        )
      .toSeq

  display(grid(rows))

/** Scores a trained model on the whole validation split: `sbt "detr/runMain detrEval"`.
  *
  * Predictions are matched to targets as in training. An object counts as detected when its
  * class is right and its defining points are within the tolerance in pixels: the two end
  * points for a part line, the anchor for a text. Four numbers are reported:
  *
  *   - `empty queries kept empty` — of the queries matched to no object, how many say so
  *     rather than inventing one. Nothing about it depends on the tolerance, so it is
  *     reported once.
  *   - `objects detected` — recall: of the objects present, how many are detected.
  *   - `detections correct` — precision: of the queries claiming an object, how many are a
  *     detection rather than a spurious or misplaced box.
  *   - `drawings fully detected` — every query slot of the drawing right at once, which
  *     includes every query beyond the objects present having to stay empty.
  *
  * The last one is all or nothing over the full query set, so it falls off as the empty-query
  * rate is raised to the power of however many queries a drawing leaves empty. That makes it
  * depend on how many queries the model has: only compare it between models of the same query
  * count, while the other three are comparable across all of them.
  */
@main
def detrEval(run: String*): Unit =
  dimwit.initialize()

  val model = load(run)
  val loss = HungarianLoss(VType[Float32])()
  val data = LShapeDetectionDataset.open(Axis[Width], Axis[Height], Axis[Channel], Axis[BoundingBox])(Split.Validation)
  val predict = jit(model.logits)

  // The split is predicted once and every tolerance scored off the same matched slots.
  val drawings = data
    .samples()
    .map: sample =>
      val prediction = predict(sample.image)
      val targets = slots(loss.matchTargets(prediction, sample.objects), data.imageWidth, data.imageHeight)
      val detected = slots(Detection(prediction.box, prediction.classLogits.argmax(Axis[ObjectClasses])), data.imageWidth, data.imageHeight)
      targets.zip(detected)
    .toSeq

  // Whether an empty query stays empty is a matter of its class alone, so any tolerance scores it.
  val empty = drawings.flatten.filter(_._1.objectClass == ObjectClass.NoObject)
  report("", "empty queries kept empty", empty.count(_._2.objectClass == ObjectClass.NoObject), empty.size)

  Tolerances.foreach: tolerance =>
    val scored = drawings.map(_.map((target, predicted) => Scored(target, predicted, isDetected(target, predicted, tolerance))))
    val objects = scored.flatten.filter(_.target.objectClass != ObjectClass.NoObject)
    val claimed = scored.flatten.filter(_.predicted.objectClass != ObjectClass.NoObject)
    report(f"$tolerance%2.0f px", "objects detected", objects.count(_.isDetected), objects.size)
    report(f"$tolerance%2.0f px", "detections correct", claimed.count(_.isDetected), claimed.size)
    report(f"$tolerance%2.0f px", "drawings fully detected", scored.count(_.forall(_.isDetected)), scored.size)

/** One scored query slot: what it should hold, what it holds, and whether that counts. */
private case class Scored(target: Slot, predicted: Slot, isDetected: Boolean)

private def report(tolerance: String, what: String, correct: Int, total: Int): Unit =
  println(f"$tolerance%5s  $what%-24s $correct%6d / $total%-6d ${100f * correct / total}%5.1f%%")

/** One query slot in pixel coordinates. */
private case class Slot(objectClass: ObjectClass, centerX: Float, centerY: Float, width: Float, height: Float):
  def left: Float = centerX - width / 2
  def right: Float = centerX + width / 2
  def top: Float = centerY - height / 2
  def bottom: Float = centerY + height / 2
  def isHorizontal: Boolean = width >= height

private def slots(detection: ObjectDetection[Float32], imageWidth: Int, imageHeight: Int): Seq[Slot] =
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

private def isDetected(target: Slot, predicted: Slot, tolerance: Float): Boolean =
  def near(expected: Float, actual: Float): Boolean = (expected - actual).abs <= tolerance
  target.objectClass == predicted.objectClass && (target.objectClass match
    case ObjectClass.NoObject => true
    case ObjectClass.Text     => near(target.centerX, predicted.centerX) && near(target.centerY, predicted.centerY)
    case ObjectClass.PartLine if target.isHorizontal =>
      near(target.left, predicted.left) && near(target.right, predicted.right) && near(target.centerY, predicted.centerY)
    case ObjectClass.PartLine =>
      near(target.top, predicted.top) && near(target.bottom, predicted.bottom) && near(target.centerX, predicted.centerX)
  )

private def load(run: Seq[String]): DETR[Float32] =
  val checkpointer = TensorTreeCheckpointer(run.headOption.getOrElse(latestRun))
  val step = checkpointer.iterations.lastOption.getOrElse(sys.error(s"no checkpoint in $CheckpointRoot"))
  println(s"loaded checkpoint $step")
  DETR(checkpointer.load[TrainState](step).get.params)

private def latestRun: String =
  Option(File(CheckpointRoot).listFiles)
    .getOrElse(Array.empty[File])
    .filter(_.isDirectory)
    .map(_.getPath)
    .maxOption
    .getOrElse(sys.error(s"no training run in $CheckpointRoot"))
