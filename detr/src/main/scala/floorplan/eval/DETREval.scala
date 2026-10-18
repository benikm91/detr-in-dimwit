package detr.floorplan.eval

import detr.*
import detr.model.*
import detr.floorplan.model.*
import detr.train.*
import detr.floorplan.train.*
import detr.floorplan.config.*
import dataset.DrawingDataset.Split
import dataset.greyLevels
import dataset.Runs
import dataset.Metrics
import dataset.RecordScoring
import dataset.Tolerances
import dataset.at
import dataset.report
import deepwit.checkpointing.TensorTreeCheckpointer
import dimwit.*
import plotwit.*
import syntheticFloorplansDataset.FloorplanDrawing
import syntheticFloorplansDataset.FloorplanGraph
import syntheticFloorplansDataset.FloorplanScoring
import syntheticFloorplansDataset.FloorplanTranscripts
import syntheticFloorplansDataset.SyntheticFloorplans
import viz.PlotTargets.websocket

/** Plots what a trained model detects: the first plans of the validation and the training split on
  * their own, with their targets, and with what the model predicts. Note that touching the training
  * split downloads its first part on first use.
  */
def plotDetector(setup: DETRSetup): Unit =
  dimwit.initialize()

  val checkpoints = TensorTreeCheckpointer.latestIn(setup.checkpointRoot).getOrElse(sys.error(s"no training run in ${setup.checkpointRoot}"))
  println(s"reading ${checkpoints.rootPath}")
  val model = DETR(checkpoints.loadLatest[TrainState].getOrElse(sys.error(s"no checkpoint in ${checkpoints.rootPath}")).params)
  val rows = Seq(Split.Validation, Split.Train).flatMap: split =>
    open(setup, split)
      .samples
      .take(3)
      .zipWithIndex
      .map: (sample, index) =>
        val drawing = greyLevels(sample.image)
        def drawn(plan: FloorplanGraph) = FloorplanDrawing(plan, drawing, Axis[Channel])
        Seq(
          plots.imagePlot(drawing, _.title := s"${split.fileName} $index"),
          plots.imagePlot(drawn(FloorplanGraph.of(sample.target).copy(edges = Seq.empty)), _.title := s"${split.fileName} $index — target"),
          plots.imagePlot(drawn(FloorplanGraph.of(model(sample.image))), _.title := s"${split.fileName} $index — predicted")
        )
      .toSeq

  display(grid(rows))

/** Scores the newest run on a whole split, and writes what it finds as [[Metrics]], and what the
  * last checkpoint found in every drawing as [[FloorplanTranscripts]]. On the validation split every
  * checkpoint is scored, the last first; the test split is looked at once, by the last alone.
  *
  * The nodes the queries answer with are compared with the record the drawing was rendered from,
  * which is how the transcribers are scored too — a detector predicts no relationships, so the
  * record it is held against holds none either. The last checkpoint is also
  * reported readably: `found` is recall, `right` is precision, and `records exactly right` is
  * every node of the drawing at once with nothing spurious.
  */
def scoreDetector(setup: DETRSetup, size: String, split: Split): Unit =
  dimwit.initialize()

  val checkpoints = TensorTreeCheckpointer.latestIn(setup.checkpointRoot).getOrElse(sys.error(s"no training run in ${setup.checkpointRoot}"))
  println(s"reading ${checkpoints.rootPath}")
  val data = open(setup, split)

  def detected(params: DETR.Params[Float32]): Seq[(FloorplanGraph, FloorplanGraph)] =
    val detect = jit(DETR(params).apply)
    data.samples
      .map(sample => (FloorplanGraph.of(sample.target).copy(edges = Seq.empty), FloorplanGraph.of(detect(sample.image))))
      .toSeq

  val csv = Metrics.Csv("detr", SyntheticFloorplans.name, size, split, Runs.parameters(checkpoints.loadLatest[TrainState].get.params), Runs.trainingSeconds(checkpoints.rootPath))
  println(s"writing to ${csv.path}")

  val scored = split match
    case Split.Test => Seq(checkpoints.iterations.last)
    case _          => checkpoints.iterations.reverse
  scored.foreach: step =>
    println(s"scoring checkpoint $step")
    val drawings = detected(checkpoints.load[TrainState](step).getOrElse(sys.error(s"no checkpoint $step")).params)
    val measured = Tolerances.map(tolerance => Metrics.Row(step, None, tolerance, drawings.map((target, found) => FloorplanScoring.score(target, found, tolerance / setup.canvas))))
    csv.append(measured)
    if step == checkpoints.iterations.last then
      measured.foreach(row => RecordScoring.reportAt(row.tolerance, row.scored))
      println(s"transcripts written to ${FloorplanTranscripts.write("detr", size, split, step, setup.canvas, drawings)}")

private[floorplan] def open(setup: DETRSetup, split: Split) =
  SyntheticFloorplans.open(Axis[Width], Axis[Height], Axis[Channel], Axis[Node], Axis[Relationship])(split, setup.canvas)
