package d2tok.floorplangraph.eval

import d2tok.*
import d2tok.floorplan.eval.Transcriber
import d2tok.floorplan.train.D2TokTrainState
import d2tok.floorplangraph.config.*
import d2tok.model.*
import dataset.DrawingDataset.Split
import dataset.Metrics
import dataset.RecordScoring
import dataset.Runs
import dataset.Tolerances
import deepwit.checkpointing.TensorTreeCheckpointer
import dimwit.*
import syntheticFloorplansDataset.FloorplanGraph
import syntheticFloorplansDataset.FloorplanScoring
import syntheticFloorplansDataset.FloorplanTranscripts
import syntheticFloorplansDataset.SyntheticFloorplans

/** Scores the newest run of a setup on a whole split, nodes and relationships, and writes the
  * metrics as one CSV, and what the last checkpoint wrote down for every plan as
  * [[FloorplanTranscripts]]. On the validation split every checkpoint is scored, the last first;
  * the test split is looked at once, by the last alone.
  */
def scoreGraphTranscriber(setup: D2TokGraphSetup, size: String, split: Split): Unit =
  dimwit.initialize()

  val checkpoints = TensorTreeCheckpointer.latestIn(setup.checkpointRoot).getOrElse(sys.error(s"no training run in ${setup.checkpointRoot}"))
  println(s"reading ${checkpoints.rootPath}")
  val data = SyntheticFloorplans.open(Axis[Width], Axis[Height], Axis[Channel], Axis[Node], Axis[Relationship])(setup.corpus, split, setup.canvas)
  println(s"${data.numSamples} plans\n")

  val transcriber = GraphTranscriber(setup, TranscribedTogether)

  /** Every plan of the split transcribed by a model, beside the record it was drawn from. */
  def transcribed(params: D2Tok.Params[Float32]): Seq[(FloorplanGraph, FloorplanGraph)] =
    data.samples
      .grouped(TranscribedTogether)
      .flatMap(batch => batch.map(sample => FloorplanGraph.of(sample.target)).zip(transcriber(params, batch.map(_.image))))
      .toSeq

  val csv = Metrics.Csv(Model, setup.corpus.name, size, split, Runs.parameters(checkpoints.loadLatest[D2TokTrainState].get.params), Runs.trainingSeconds(checkpoints.rootPath))
  println(s"writing to ${csv.path}")

  val scored = split match
    case Split.Test => Seq(checkpoints.iterations.last)
    case _          => checkpoints.iterations.reverse
  scored.foreach: step =>
    println(s"scoring checkpoint $step")
    val plans = transcribed(checkpoints.load[D2TokTrainState](step).getOrElse(sys.error(s"no checkpoint $step")).params)
    val measured = Tolerances.map(tolerance => Metrics.Row(step, None, tolerance, plans.map((target, written) => FloorplanScoring.score(target, written, tolerance / setup.canvas))))
    csv.append(measured)
    if step == checkpoints.iterations.last then
      measured.foreach(row => RecordScoring.reportAt(row.tolerance, row.scored))
      println(s"transcripts written to ${FloorplanTranscripts.write(Model, setup.corpus, size, split, step, setup.canvas, plans)}")

/** What the metrics, transcripts and pictures are filed under. */
val Model = "d2tok-graph"

/** Writes a plan's nodes and relationships in one sequence, as [[Transcriber]] writes its nodes,
  * and reads the words back as a graph.
  */
class GraphTranscriber(setup: D2TokGraphSetup, drawings: Int):

  private val transcriber = Transcriber(setup.words.nodeWords, Axis[Token] -> setup.tokens, drawings)

  def apply(params: D2Tok.Params[Float32], documents: Seq[Tensor3[Width, Height, Channel, Float32]]): Seq[FloorplanGraph] =
    transcriber.written(params, documents).map(setup.words.read)

/** How many plans are transcribed together. A batch is one traced computation, so this is what the
  * split costs in calls rather than in work.
  */
private val TranscribedTogether = 32
