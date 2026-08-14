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

/** How far an object's defining points may be off, in pixels. */
private val Tolerance = 4f

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
  * class is right and its defining points are within [[Tolerance]] pixels: the two end points
  * for a part line, the anchor for a text. A drawing counts as detected when every one of its
  * query slots is right — every object detected, none missing and none spurious.
  */
@main
def detrEval(run: String*): Unit =
  dimwit.initialize()

  val model = load(run)
  val loss = HungarianLoss(VType[Float32])()
  val data = LShapeDetectionDataset.open(Axis[Width], Axis[Height], Axis[Channel], Axis[BoundingBox])(Split.Validation)
  val predict = jit(model.logits)

  val drawings = data
    .samples()
    .map: sample =>
      val prediction = predict(sample.image)
      val targets = slots(loss.matchTargets(prediction, sample.objects), data.imageWidth, data.imageHeight)
      val detected = slots(Detection(prediction.box, prediction.classLogits.argmax(Axis[ObjectClasses])), data.imageWidth, data.imageHeight)
      targets.zip(detected).map((target, prediction) => (target, isDetected(target, prediction)))
    .toSeq

  val objects = drawings.flatten.filter(_._1.objectClass != ObjectClass.NoObject)
  report("objects detected", objects.count(_._2), objects.size)
  report("drawings fully detected", drawings.count(_.forall(_._2)), drawings.size)

private def report(what: String, correct: Int, total: Int): Unit =
  println(f"$what%-24s $correct%6d / $total%-6d ${100f * correct / total}%5.1f%%")

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

private def isDetected(target: Slot, predicted: Slot): Boolean =
  def near(expected: Float, actual: Float): Boolean = (expected - actual).abs <= Tolerance
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
