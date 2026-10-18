package detr.floorplan.eval

import detr.*
import detr.floorplan.config.*
import detr.floorplan.model.*
import detr.floorplan.train.*
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

/** Writes what every checkpoint of the newest run detected, one picture each:
  * `sbt "detr/runMain detrDraw synthetic-floorplans s-deep"`.
  *
  * Every validation plan is shown as it was drawn, beside what the model detected on a canvas
  * of its own. The pictures land beside the metrics in `OUTPUT_DIR`.
  */
def drawDetector(setup: DETRSetup, size: String): Unit =
  dimwit.initialize()

  val checkpoints = TensorTreeCheckpointer.latestIn(setup.checkpointRoot).getOrElse(sys.error(s"no training run in ${setup.checkpointRoot}"))
  println(s"reading ${checkpoints.rootPath}")

  val drawings = open(setup, Split.Validation).samples.take(Rows * Across).toSeq

  /** The drawings as they were drawn, read once since they do not change. */
  val documents = drawings.map(sample => greyLevels(sample.image))

  /** An empty canvas holds a detection on its own; an empty record leaves a drawing as it is. */
  val emptyCanvas = Tensor.like(documents.head).fill(Blank)
  val noRecord = FloorplanGraph(Seq.empty, Seq.empty)

  for step <- checkpoints.iterations do
    val detect = jit(DETR(checkpoints.load[TrainState](step).getOrElse(sys.error(s"no checkpoint $step")).params).apply)
    val eachDrawing = drawings.zip(documents).map: (sample, document) =>
      Seq(FloorplanDrawing(noRecord, document, Axis[Channel]), FloorplanDrawing(FloorplanGraph.of(detect(sample.image)), emptyCanvas, Axis[Channel]))
    val picture = Path.of(Runs.outputDir, s"detr-${SyntheticFloorplans.name}-$size-$step.png")
    DrawingTiles.write(eachDrawing, Rows, Across, picture)
    println(s"step $step | ${drawings.size} drawings written to $picture")

/** The picture: drawings across, rows down, each drawing beside what the model made of it. */
private val Rows = 8
private val Across = 8

/** An empty canvas, as [[dataset.greyLevels]] gives one. */
private val Blank = 255
