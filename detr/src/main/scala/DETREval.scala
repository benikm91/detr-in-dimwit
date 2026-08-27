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
    val data = LShapeDataset.open(Axis[Width], Axis[Height], Axis[Channel], Axis[BoundingBox], Axis[Relationship])(split)
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
  * The lines are the ones [[RecordScoring.reportAt]] reports for every model — `found` is recall,
  * `right` is precision, and `records exactly right` is every node of the drawing at once with
  * nothing spurious. A detector predicts no relationships, so those lines are left out.
  */
@main
def detrEval(): Unit =
  dimwit.initialize()

  val checkpoints = TensorTreeCheckpointer.latestIn(CheckpointRoot).getOrElse(sys.error(s"no training run in $CheckpointRoot"))
  println(s"reading ${checkpoints.rootPath}")
  val model = DETR(checkpoints.loadLatest[TrainState].getOrElse(sys.error(s"no checkpoint in ${checkpoints.rootPath}")).params)
  val data = LShapeDataset.open(Axis[Width], Axis[Height], Axis[Channel], Axis[BoundingBox], Axis[Relationship])(Split.Validation)
  val detect = jit(model.apply)

  val drawings = data
    .samples
    .map: sample =>
      (RecordGraph.of(sample.target).copy(edges = Seq.empty), RecordGraph.of(detect(sample.image)))
    .toSeq

  Tolerances.foreach: tolerance =>
    RecordScoring.reportAt(tolerance, drawings.map((target, detected) => RecordScoring.score(target, detected, tolerance / Canvas)))
