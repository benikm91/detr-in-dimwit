package detr.eval

import detr.*
import detr.model.*
import detr.train.*
import detr.config.*
import dataset.Corpus
import dataset.DrawingDataset
import dataset.DrawingDataset.Split
import dataset.greyLevels
import dataset.RecordDrawing
import dataset.RecordGraph
import dataset.Runs
import dataset.Metrics
import dataset.RecordScoring
import dataset.Tolerances
import dataset.Transcripts
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
    val data = DrawingDataset.open(setup.corpus)(Axis[Width], Axis[Height], Axis[Channel], Axis[Node], Axis[Relationship])(split)
    data
      .samples
      .take(3)
      .zipWithIndex
      .map: (sample, index) =>
        val drawing = greyLevels(sample.image)
        def drawn(record: RecordGraph) = RecordDrawing(record, drawing, Axis[Channel])
        Seq(
          plots.imagePlot(drawing, _.title := s"${split.fileName} $index"),
          plots.imagePlot(drawn(RecordGraph.of(sample.target).copy(edges = Seq.empty)), _.title := s"${split.fileName} $index — target"),
          plots.imagePlot(drawn(RecordGraph.of(model(sample.image))), _.title := s"${split.fileName} $index — predicted")
        )
      .toSeq

  display(grid(rows))

/** Scores every checkpoint of the newest run on the whole validation split, the last checkpoint
  * first, and writes what it finds as [[Metrics]], and what the last checkpoint found in every
  * drawing as [[Transcripts]].
  *
  * The nodes the queries answer with are compared with the record the drawing was rendered from,
  * which is how [[scoreTranscriber]] scores too — a detector predicts no relationships, so the
  * record it is held against holds none either. The last checkpoint is also
  * reported readably: `found` is recall, `right` is precision, and `records exactly right` is
  * every node of the drawing at once with nothing spurious.
  */
def scoreDetector(setup: DETRSetup, size: String): Unit =
  dimwit.initialize()

  val checkpoints = TensorTreeCheckpointer.latestIn(setup.checkpointRoot).getOrElse(sys.error(s"no training run in ${setup.checkpointRoot}"))
  println(s"reading ${checkpoints.rootPath}")
  val data = DrawingDataset.open(setup.corpus)(Axis[Width], Axis[Height], Axis[Channel], Axis[Node], Axis[Relationship])(Split.Validation)

  def detected(params: DETR.Params[Float32]): Seq[(RecordGraph, RecordGraph)] =
    val detect = jit(DETR(params).apply)
    data.samples
      .map(sample => (RecordGraph.of(sample.target).copy(edges = Seq.empty), RecordGraph.of(detect(sample.image))))
      .toSeq

  val csv = Metrics.Csv("detr", setup.corpus, size, Runs.parameters(checkpoints.loadLatest[TrainState].get.params), Runs.trainingSeconds(checkpoints.rootPath))
  println(s"writing to ${csv.path}")

  checkpoints.iterations.reverse.foreach: step =>
    println(s"scoring checkpoint $step")
    val drawings = detected(checkpoints.load[TrainState](step).getOrElse(sys.error(s"no checkpoint $step")).params)
    val measured = Tolerances.map(tolerance => Metrics.Row(step, None, tolerance, drawings.map((target, found) => RecordScoring.score(target, found, tolerance / setup.corpus.canvas))))
    csv.append(measured)
    if step == checkpoints.iterations.last then
      measured.foreach(row => RecordScoring.reportAt(row.tolerance, row.scored))
      println(s"transcripts written to ${Transcripts.write("detr", setup.corpus, size, step, drawings)}")
