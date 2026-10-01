import detr.*
import detr.config.*
import detr.model.*
import detr.train.*
import dataset.Corpus
import dataset.DrawingDataset
import dataset.DrawingDataset.Split
import dataset.DrawingTiles
import dataset.Outlines
import dataset.RecordDrawing
import dataset.RecordGraph
import dataset.Runs
import deepwit.checkpointing.TensorTreeCheckpointer
import dimwit.*

import java.nio.file.Path

/** Writes what every checkpoint of the newest run detected, one picture each:
  * `sbt "detr/runMain detrDraw l-shape s"`.
  *
  * Every validation drawing is shown as it was drawn, beside what the model detected on a canvas
  * of its own. The pictures land beside the metrics in `OUTPUT_DIR`.
  */
@main
def detrDraw(corpus: String, size: String): Unit =
  dimwit.initialize()
  val setup = DETRSetup(Corpus.named(corpus), DETR.Size.named(size))

  val checkpoints = TensorTreeCheckpointer.latestIn(setup.checkpointRoot).getOrElse(sys.error(s"no training run in ${setup.checkpointRoot}"))
  println(s"reading ${checkpoints.rootPath}")

  val data = DrawingDataset.open(setup.corpus)(Axis[Width], Axis[Height], Axis[Channel], Axis[Node], Axis[Relationship])(Split.Validation)
  val drawings = data.samples.take(Rows * Across).toSeq

  /** The drawings as they were drawn, read once since they do not change. */
  val documents = drawings.map(sample => Outlines.greyLevels(sample.image))

  /** An empty canvas holds a detection on its own; an empty record leaves a drawing as it is. */
  val emptyCanvas = Tensor.like(documents.head).fill(Blank)
  val noRecord = RecordGraph(Seq.empty, Seq.empty)

  for step <- checkpoints.iterations do
    val detect = jit(DETR(checkpoints.load[TrainState](step).getOrElse(sys.error(s"no checkpoint $step")).params).apply)
    val eachDrawing = drawings.zip(documents).map: (sample, document) =>
      Seq(RecordDrawing(noRecord, document, Axis[Channel]), RecordDrawing(RecordGraph.of(detect(sample.image)), emptyCanvas, Axis[Channel]))
    val picture = Path.of(Runs.outputDir, s"detr-${setup.corpus.name}-$size-$step.png")
    DrawingTiles.write(eachDrawing, Rows, Across, picture)
    println(s"step $step | ${drawings.size} drawings written to $picture")

/** The picture: drawings across, rows down, each drawing beside what the model made of it. */
private val Rows = 16
private val Across = 16

/** An empty canvas, as [[dataset.Outlines]] reads one. */
private val Blank = 255
