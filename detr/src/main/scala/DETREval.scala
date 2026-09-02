package detr

import dataset.Canvas
import dataset.Corpus
import dataset.DrawingDataset
import dataset.DrawingDataset.Split
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

/** Plots what the l-shape run detects: `sbt "detr/runMain detr.detrLShapePlot"`. */
@main
def detrLShapePlot(): Unit =
  import detr.lshape.{CheckpointRoot, TrainState}
  dimwit.initialize()

  val checkpoints = TensorTreeCheckpointer.latestIn(CheckpointRoot).getOrElse(sys.error(s"no training run in $CheckpointRoot"))
  println(s"reading ${checkpoints.rootPath}")
  val model = DETR(checkpoints.loadLatest[TrainState].getOrElse(sys.error(s"no checkpoint in ${checkpoints.rootPath}")).params)
  detrPlot(Corpus.LShape, model)

/** Scores the l-shape run: `sbt "detr/runMain detr.detrLShapeEval"`. */
@main
def detrLShapeEval(): Unit =
  import detr.lshape.{CheckpointRoot, TrainState}
  dimwit.initialize()

  val checkpoints = TensorTreeCheckpointer.latestIn(CheckpointRoot).getOrElse(sys.error(s"no training run in $CheckpointRoot"))
  println(s"reading ${checkpoints.rootPath}")
  val model = DETR(checkpoints.loadLatest[TrainState].getOrElse(sys.error(s"no checkpoint in ${checkpoints.rootPath}")).params)
  detrEval(Corpus.LShape, model)

/** Plots what the rectilinear run detects: `sbt "detr/runMain detr.detrRectilinear6to18Plot"`. */
@main
def detrRectilinear6to18Plot(): Unit =
  import detr.rectilinear6to18.{CheckpointRoot, TrainState}
  dimwit.initialize()

  val checkpoints = TensorTreeCheckpointer.latestIn(CheckpointRoot).getOrElse(sys.error(s"no training run in $CheckpointRoot"))
  println(s"reading ${checkpoints.rootPath}")
  val model = DETR(checkpoints.loadLatest[TrainState].getOrElse(sys.error(s"no checkpoint in ${checkpoints.rootPath}")).params)
  detrPlot(Corpus.Rectilinear6to18, model)

/** Scores the rectilinear run: `sbt "detr/runMain detr.detrRectilinear6to18Eval"`. */
@main
def detrRectilinear6to18Eval(): Unit =
  import detr.rectilinear6to18.{CheckpointRoot, TrainState}
  dimwit.initialize()

  val checkpoints = TensorTreeCheckpointer.latestIn(CheckpointRoot).getOrElse(sys.error(s"no training run in $CheckpointRoot"))
  println(s"reading ${checkpoints.rootPath}")
  val model = DETR(checkpoints.loadLatest[TrainState].getOrElse(sys.error(s"no checkpoint in ${checkpoints.rootPath}")).params)
  detrEval(Corpus.Rectilinear6to18, model)

/** Plots what a trained model detects.
  *
  * Shows the first drawings of the validation and the training split on their own, with their
  * targets, and with what the model predicts. Note that touching the training split downloads the
  * whole of it on first use.
  */
private def detrPlot(corpus: Corpus, model: DETR[Float32]): Unit =
  val rows = Seq(Split.Validation, Split.Train).flatMap: split =>
    val data = DrawingDataset.open(corpus)(Axis[Width], Axis[Height], Axis[Channel], Axis[BoundingBox], Axis[Relationship])(split)
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

/** Scores a trained model on the whole validation split of a corpus.
  *
  * What is detected is read back into the record it stands for and compared with the record the
  * drawing was rendered from, which is how [[d2gEval]] scores too — a detector predicts no
  * relationships, so the record it is held against holds none either.
  *
  * The lines are the ones [[RecordScoring.reportAt]] reports for every model — `found` is recall,
  * `right` is precision, and `records exactly right` is every node of the drawing at once with
  * nothing spurious. A detector predicts no relationships, so those lines are left out.
  */
private def detrEval(corpus: Corpus, model: DETR[Float32]): Unit =
  val data = DrawingDataset.open(corpus)(Axis[Width], Axis[Height], Axis[Channel], Axis[BoundingBox], Axis[Relationship])(Split.Validation)
  val detect = jit(model.apply)

  val drawings = data
    .samples
    .map: sample =>
      (RecordGraph.of(sample.target).copy(edges = Seq.empty), RecordGraph.of(detect(sample.image)))
    .toSeq

  Tolerances.foreach: tolerance =>
    RecordScoring.reportAt(tolerance, drawings.map((target, detected) => RecordScoring.score(target, detected, tolerance / Canvas)))
