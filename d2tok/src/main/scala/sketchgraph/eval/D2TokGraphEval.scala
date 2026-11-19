package d2tok.sketchgraph.eval

import d2tok.*
import d2tok.model.*
import d2tok.sketch.eval.Transcriber
import d2tok.sketch.train.D2TokTrainState
import d2tok.sketchgraph.config.*
import dataset.DrawingDataset
import dataset.DrawingDataset.Split
import dataset.Metrics
import dataset.RecordGraph
import dataset.RecordScoring
import dataset.Runs
import dataset.Tolerances
import dataset.Transcripts
import deepwit.checkpointing.TensorTreeCheckpointer
import dimwit.*

/** Scores the newest run of a setup on a whole split, nodes and relationships, and writes the
  * metrics as one CSV, and what the last checkpoint wrote down for every drawing as [[Transcripts]].
  * On the validation split every checkpoint is scored, the last first; the test split is looked at
  * once, by the last alone.
  */
def scoreGraphTranscriber(setup: D2TokGraphSetup, size: String, split: Split): Unit =
  dimwit.initialize()

  val checkpoints = TensorTreeCheckpointer.latestIn(setup.checkpointRoot).getOrElse(sys.error(s"no training run in ${setup.checkpointRoot}"))
  println(s"reading ${checkpoints.rootPath}")
  val data = DrawingDataset.open(setup.corpus)(Axis[Width], Axis[Height], Axis[Channel], Axis[Node], Axis[Relationship])(split)
  println(s"${data.numSamples} drawings\n")

  val transcriber = Transcriber(setup.words.nodeWords, Axis[Token] -> setup.tokens, TranscribedTogether)

  /** Every drawing of the split transcribed by a model, beside the record it was rendered from. */
  def transcribed(params: D2Tok.Params[Float32]): Seq[(RecordGraph, RecordGraph)] =
    data.samples
      .grouped(TranscribedTogether)
      .flatMap(batch => batch.map(sample => RecordGraph.of(sample.target)).zip(transcriber.written(params, batch.map(_.image)).map(setup.words.read)))
      .toSeq

  val csv = Metrics.Csv(Model, setup.corpus.name, size, split, Runs.parameters(checkpoints.loadLatest[D2TokTrainState].get.params), Runs.trainingSeconds(checkpoints.rootPath))
  println(s"writing to ${csv.path}")

  val scored = split match
    case Split.Test => Seq(checkpoints.iterations.last)
    case _          => checkpoints.iterations.reverse
  scored.foreach: step =>
    println(s"scoring checkpoint $step")
    val drawings = transcribed(checkpoints.load[D2TokTrainState](step).getOrElse(sys.error(s"no checkpoint $step")).params)
    val measured = Tolerances.map(tolerance => Metrics.Row(step, None, tolerance, drawings.map((target, written) => RecordScoring.score(target, written, tolerance / setup.corpus.canvas))))
    csv.append(measured)
    if step == checkpoints.iterations.last then
      measured.foreach(row => RecordScoring.reportAt(row.tolerance, row.scored))
      println(s"transcripts written to ${Transcripts.write(Model, setup.corpus, size, split, step, drawings)}")

/** What the metrics and transcripts are filed under. */
private val Model = "d2tok-graph"

/** How many drawings are transcribed together. A batch is one traced computation, so this is what
  * the split costs in calls rather than in work.
  */
private val TranscribedTogether = 32
