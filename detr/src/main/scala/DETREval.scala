import dataset.Canvas
import dataset.LShapeDataset
import dataset.LShapeDataset.Split
import dataset.Outlines
import dataset.RecordGraph
import dataset.RecordScoring
import dataset.Tolerances
import dataset.at
import dataset.report
import deepwit.checkpointing.TensorTreeCheckpointer
import dimwit.*
import plotwit.*
import viz.PlotTargets.websocket

/** Plots what a trained model detects: `sbt "detr/runMain detrPlot"`.
  *
  * Shows the first drawings of the validation and the training split on their own, with their
  * targets, and with what the model predicts. Note that touching the training split downloads
  * 8.6 GB on first use.
  */
@main
def detrPlot(): Unit =
  dimwit.initialize()

  val checkpoints = TensorTreeCheckpointer.latestIn(CheckpointRoot).getOrElse(sys.error(s"no training run in $CheckpointRoot"))
  println(s"reading ${checkpoints.rootPath}")
  val model = DETR(checkpoints.loadLatest[TrainState].getOrElse(sys.error(s"no checkpoint in ${checkpoints.rootPath}")).params)
  val rows = Seq(Split.Validation, Split.Train).flatMap: split =>
    val data = LShapeDataset.open(Axis[Width], Axis[Height], Axis[Channel], Axis[BoundingBox])(split)
    data
      .objects
      .take(3)
      .zipWithIndex
      .map: (sample, index) =>
        val drawing = Outlines.greyLevels(sample.image)
        Seq(
          plots.imagePlot(drawing, _.title := s"${split.fileName} $index"),
          plots.imagePlot(Outlines(drawing, sample.target.detection), _.title := s"${split.fileName} $index — target"),
          plots.imagePlot(Outlines(drawing, model(sample.image)), _.title := s"${split.fileName} $index — predicted")
        )
      .toSeq

  display(grid(rows))

/** Scores a trained model on the whole validation split: `sbt "detr/runMain detrEval"`.
  *
  * What is detected is read back into the record it stands for and compared with the record the
  * drawing was rendered from, which is how [[d2gEval]] scores too — a detector predicts no
  * relationships, so the record it is held against holds none either.
  *
  *   - `nodes detected` — recall: of the lines and annotations of the split, how many the model
  *     found, to within the tolerance.
  *   - `detections right` — precision: of the objects it claims, how many are one.
  *   - `drawings fully detected` — every node of the drawing found at once, and nothing spurious.
  */
@main
def detrEval(): Unit =
  dimwit.initialize()

  val checkpoints = TensorTreeCheckpointer.latestIn(CheckpointRoot).getOrElse(sys.error(s"no training run in $CheckpointRoot"))
  println(s"reading ${checkpoints.rootPath}")
  val model = DETR(checkpoints.loadLatest[TrainState].getOrElse(sys.error(s"no checkpoint in ${checkpoints.rootPath}")).params)
  val data = LShapeDataset.open(Axis[Width], Axis[Height], Axis[Channel], Axis[BoundingBox])(Split.Validation)
  val detect = jit(model.apply)

  val drawings = data
    .samples
    .map: sample =>
      (RecordGraph.of(sample.target).copy(edges = Seq.empty), RecordGraph.of(detect(sample.image)))
    .toSeq

  Tolerances.foreach: tolerance =>
    val scored = drawings.map((target, detected) => RecordScoring.score(target, detected, tolerance / Canvas))
    report(at(tolerance), "nodes detected", scored.map(_.nodesFound).sum, scored.map(_.nodes).sum)
    report(at(tolerance), "detections right", scored.map(_.nodesFound).sum, scored.map(_.nodesPredicted).sum)
    report(at(tolerance), "drawings fully detected", scored.count(_.isExact), scored.length)
