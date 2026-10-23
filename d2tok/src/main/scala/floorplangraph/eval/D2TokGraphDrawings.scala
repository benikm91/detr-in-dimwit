package d2tok.floorplangraph.eval

import d2tok.*
import d2tok.floorplangraph.config.*
import d2tok.floorplangraph.train.D2TokGraphTrainState
import dataset.DrawingDataset.Split
import dataset.DrawingTiles
import dataset.greyLevels
import dataset.Runs
import deepwit.checkpointing.TensorTreeCheckpointer
import dimwit.*
import syntheticFloorplansDataset.FloorplanDrawing
import syntheticFloorplansDataset.FloorplanGraph
import syntheticFloorplansDataset.SyntheticFloorplans

import java.nio.file.Path

/** Writes what every checkpoint of the newest run transcribed, one picture each:
  * `sbt "d2tok/runMain d2tokGraphDraw s-deep"`.
  *
  * Every validation plan is shown as it was drawn, beside the record the model wrote down on a
  * canvas of its own. The pictures land beside the metrics in `OUTPUT_DIR`.
  */
def drawGraphTranscriber(setup: D2TokGraphSetup, size: String): Unit =
  dimwit.initialize()

  val checkpoints = TensorTreeCheckpointer.latestIn(setup.checkpointRoot).getOrElse(sys.error(s"no training run in ${setup.checkpointRoot}"))
  println(s"reading ${checkpoints.rootPath}")

  val plans = SyntheticFloorplans.open(Axis[Width], Axis[Height], Axis[Channel], Axis[Node], Axis[Relationship])(Split.Validation, setup.canvas).samples.take(Rows * Across).toSeq
  val transcriber = Transcriber(setup, WrittenTogether)

  /** The plans as they were drawn, read once since they do not change. */
  val documents = plans.map(sample => greyLevels(sample.image))

  /** An empty canvas holds a transcription on its own; an empty record leaves a plan as it is. */
  val emptyCanvas = Tensor.like(documents.head).fill(Blank)
  val noRecord = FloorplanGraph(Seq.empty, Seq.empty)

  for step <- checkpoints.iterations do
    val params = checkpoints.load[D2TokGraphTrainState](step).getOrElse(sys.error(s"no checkpoint $step")).params
    val written = plans.grouped(WrittenTogether).flatMap(batch => transcriber(params, batch.map(_.image))).toSeq
    val eachPlan = documents.zip(written).map: (document, transcribed) =>
      Seq(FloorplanDrawing(noRecord, document, Axis[Channel]), FloorplanDrawing(transcribed, emptyCanvas, Axis[Channel]))
    val picture = Path.of(Runs.outputDir, s"$Model-${SyntheticFloorplans.name}-$size-$step.png")
    DrawingTiles.write(eachPlan, Rows, Across, picture)
    println(s"step $step | ${written.size} plans written to $picture")

/** The picture: plans across, rows down, each plan beside what the model made of it. */
private val Rows = 8
private val Across = 8

/** An empty canvas, as [[dataset.greyLevels]] gives one. */
private val Blank = 255

/** How many plans are transcribed together, as in the scoring: one traced computation. */
private val WrittenTogether = 32
