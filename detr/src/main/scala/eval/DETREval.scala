package detr.eval

import detr.*
import detr.model.*
import detr.train.*
import detr.config.*
import dataset.Canvas
import dataset.Corpus
import dataset.DrawingDataset
import dataset.DrawingDataset.Split
import dataset.Outlines
import dataset.RecordGraph
import dataset.Runs
import dataset.Metrics
import dataset.RecordScoring
import dataset.Tolerances
import dataset.at
import dataset.report
import deepwit.checkpointing.TensorTreeCheckpointer
import dimwit.*
import plotwit.*
import viz.PlotTargets.websocket

/** Plots what a trained model detects.
  *
  * Shows the first drawings of the validation and the training split on their own, with their
  * targets, and with what the model predicts. Note that touching the training split downloads the
  * whole of it on first use.
  */
def plotDetector(setup: DETRSetup): Unit =
  dimwit.initialize()

  val checkpoints = TensorTreeCheckpointer.latestIn(setup.checkpointRoot).getOrElse(sys.error(s"no training run in ${setup.checkpointRoot}"))
  println(s"reading ${checkpoints.rootPath}")
  val model = DETR(checkpoints.loadLatest[TrainState].getOrElse(sys.error(s"no checkpoint in ${checkpoints.rootPath}")).params)
  val rows = Seq(Split.Validation, Split.Train).flatMap: split =>
    val data = DrawingDataset.open(setup.corpus)(Axis[Width], Axis[Height], Axis[Channel], Axis[BoundingBox], Axis[Relationship])(split)
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

/** Scores every checkpoint of the newest run on the whole validation split, and writes what it
  * finds as [[Metrics]].
  *
  * What is detected is read back into the record it stands for and compared with the record the
  * drawing was rendered from, which is how [[scoreTranscriber]] scores too — a detector predicts no
  * relationships, so the record it is held against holds none either. The last checkpoint is also
  * reported readably: `found` is recall, `right` is precision, and `records exactly right` is
  * every node of the drawing at once with nothing spurious.
  */
def scoreDetector(setup: DETRSetup, size: String): Unit =
  dimwit.initialize()

  val checkpoints = TensorTreeCheckpointer.latestIn(setup.checkpointRoot).getOrElse(sys.error(s"no training run in ${setup.checkpointRoot}"))
  println(s"reading ${checkpoints.rootPath}")
  val data = DrawingDataset.open(setup.corpus)(Axis[Width], Axis[Height], Axis[Channel], Axis[BoundingBox], Axis[Relationship])(Split.Validation)

  def detected(params: DETR.Params[Float32]): Seq[(RecordGraph, RecordGraph)] =
    val detect = jit(DETR(params).apply)
    data.samples
      .map(sample => (RecordGraph.of(sample.target).copy(edges = Seq.empty), RecordGraph.of(detect(sample.image))))
      .toSeq

  val rows = checkpoints.iterations.flatMap: step =>
    println(s"scoring checkpoint $step")
    val drawings = detected(checkpoints.load[TrainState](step).getOrElse(sys.error(s"no checkpoint $step")).params)
    Tolerances.map(tolerance => Metrics.Row(step, None, tolerance, drawings.map((target, found) => RecordScoring.score(target, found, tolerance / Canvas))))

  rows.filter(_.step == checkpoints.iterations.last).foreach(row => RecordScoring.reportAt(row.tolerance, row.scored))
  val params = checkpoints.loadLatest[TrainState].get.params
  println(s"written to ${Metrics.write("detr", setup.corpus, size, Runs.parameters(params), Runs.trainingSeconds(checkpoints.rootPath), rows)}")
