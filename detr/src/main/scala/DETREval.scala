import dataset.Detection
import dataset.LShapeDetectionDataset
import dataset.LShapeDetectionDataset.Split
import dataset.ObjectClass
import dataset.Outlines
import dimwit.*
import plotwit.*
import viz.PlotTargets.websocket

import DetectionScoring.Slot
import DetectionScoring.at
import DetectionScoring.isDetected
import DetectionScoring.report
import DetectionScoring.slots

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
  val empty = drawings.flatten.filter(!_._1.isObject)
  report("", "empty queries kept empty", empty.count(!_._2.isObject), empty.size)

  Tolerances.foreach: tolerance =>
    val scored = drawings.map(_.map((target, predicted) => Scored(target, predicted, isDetected(target, predicted, tolerance))))
    val objects = scored.flatten.filter(_.target.isObject)
    val claimed = scored.flatten.filter(_.predicted.isObject)
    report(at(tolerance), "objects detected", objects.count(_.isDetected), objects.size)
    report(at(tolerance), "detections correct", claimed.count(_.isDetected), claimed.size)
    report(at(tolerance), "drawings fully detected", scored.count(_.forall(_.isDetected)), scored.size)

/** One scored query slot: what it should hold, what it holds, and whether that counts. */
private case class Scored(target: Slot, predicted: Slot, isDetected: Boolean)

private def load(run: Seq[String]): DETR[Float32] =
  DETR(Checkpoints.loadLatest[TrainState](CheckpointRoot, run).params)
