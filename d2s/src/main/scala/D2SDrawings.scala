import d2s.*
import d2s.config.*
import d2s.model.*
import d2s.eval.Transcriber
import d2s.train.D2STrainState
import dataset.Corpus
import dataset.DrawingDataset
import dataset.DrawingDataset.Split
import dataset.DrawingTiles
import dataset.greyLevels
import dataset.RecordDrawing
import dataset.RecordGraph
import dataset.Runs
import deepwit.checkpointing.TensorTreeCheckpointer
import dimwit.*

import java.nio.file.Path

/** Writes what every checkpoint of the newest run transcribed, one picture each:
  * `sbt "d2s/runMain d2sDraw sketch s-deep"`.
  *
  * A score says a transcription is wrong without saying how, and where the records themselves are
  * uncertain — one stroke written down as two — how is most of what there is to know. Every
  * validation drawing is shown as it was drawn, beside the record the model wrote down on a canvas
  * of its own. The pictures land beside the metrics in `OUTPUT_DIR`.
  */
@main
def d2sDraw(corpus: String, size: String): Unit =
  dimwit.initialize()
  val setup = D2SSetup(Corpus.named(corpus), D2SModelConfiguration.named(size))

  val checkpoints = TensorTreeCheckpointer.latestIn(setup.checkpointRoot).getOrElse(sys.error(s"no training run in ${setup.checkpointRoot}"))
  println(s"reading ${checkpoints.rootPath}")

  val data = DrawingDataset.open(setup.corpus)(Axis[Width], Axis[Height], Axis[Channel], Axis[Node], Axis[Relationship])(Split.Validation)
  val drawings = data.samples.take(Rows * Across).toSeq
  val transcriber = Transcriber(Axis[Node] -> setup.nodeSlots, WrittenTogether)

  /** The drawings as they were drawn, read once since they do not change. */
  val documents = drawings.map(sample => greyLevels(sample.image))

  /** An empty canvas holds a transcription on its own; an empty record leaves a drawing as it is. */
  val emptyCanvas = Tensor.like(documents.head).fill(Blank)
  val noRecord = RecordGraph(Seq.empty, Seq.empty)

  for step <- checkpoints.iterations do
    val params = checkpoints.load[D2STrainState](step).getOrElse(sys.error(s"no checkpoint $step")).params
    val written = drawings.grouped(WrittenTogether).flatMap(batch => transcriber(params, batch.map(_.image))).toSeq
    val eachDrawing = documents.zip(written).map: (document, transcribed) =>
      Seq(RecordDrawing(noRecord, document, Axis[Channel]), RecordDrawing(transcribed, emptyCanvas, Axis[Channel]))
    val picture = Path.of(Runs.outputDir, s"d2s-${setup.corpus.name}-$size-$step.png")
    DrawingTiles.write(eachDrawing, Rows, Across, picture)
    println(s"step $step | ${written.size} drawings written to $picture")

/** The picture: drawings across, rows down, each drawing beside what the model made of it. */
private val Rows = 16
private val Across = 16

/** An empty canvas, as [[dataset.greyLevels]] gives one. */
private val Blank = 255

/** How many drawings are transcribed together, as in the scoring: one traced computation. */
private val WrittenTogether = 32
