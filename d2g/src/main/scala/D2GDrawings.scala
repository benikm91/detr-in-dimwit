import d2g.*
import d2g.config.*
import d2g.eval.Transcriber
import d2g.train.D2GTrainState
import dataset.Corpus
import dataset.DrawingDataset
import dataset.DrawingDataset.Split
import dataset.Outlines
import dataset.RecordDrawing
import dataset.RecordGraph
import dataset.Runs
import deepwit.checkpointing.TensorTreeCheckpointer
import dimwit.*
import dimwit.jax.Jax
import dimwit.python.PyBridge.toPyTensor
import me.shadaj.scalapy.py

import java.nio.file.Files
import java.nio.file.Path
import java.nio.file.StandardCopyOption

/** Writes what every checkpoint of the newest run transcribed, one picture each:
  * `sbt "d2g/runMain d2gDraw sketch s-deep"`.
  *
  * A score says a transcription is wrong without saying how, and where the records themselves are
  * uncertain — one stroke written down as two — how is most of what there is to know. Every
  * validation drawing is shown three times: the record it was rendered from, the record the model
  * wrote down over the same drawing, and that record on its own. The pictures land beside the
  * metrics in `OUTPUT_DIR`.
  */
@main
def d2gDraw(corpus: String, size: String): Unit =
  dimwit.initialize()
  val setup = D2GSetup(Corpus.named(corpus), D2GModelConfiguration.named(size))

  val checkpoints = TensorTreeCheckpointer.latestIn(setup.checkpointRoot).getOrElse(sys.error(s"no training run in ${setup.checkpointRoot}"))
  println(s"reading ${checkpoints.rootPath}")

  val data = DrawingDataset.open(setup.corpus)(Axis[Width], Axis[Height], Axis[Channel], Axis[Node], Axis[Edge])(Split.Validation)
  val drawings = data.samples.take(Rows * Across).toSeq
  val transcriber = Transcriber(Axis[Node] -> setup.nodeSlots, Axis[Edge] -> setup.edgeSlots, WrittenTogether)

  /** What every checkpoint is drawn over and held against, read once since it does not change. */
  val documents = drawings.map(sample => Outlines.greyLevels(sample.image))
  val targets = drawings.map(sample => RecordGraph.of(sample.target))

  /** The same canvas with nothing on it, so a transcription can also be read on its own. */
  val nothing = Tensor.like(documents.head).fill(Blank)

  for step <- checkpoints.iterations do
    val params = checkpoints.load[D2GTrainState](step).getOrElse(sys.error(s"no checkpoint $step")).params
    val written = drawings.grouped(WrittenTogether).flatMap(batch => transcriber(params, batch.map(_.image))).toSeq
    val eachDrawing = documents.lazyZip(targets).lazyZip(written).map: (document, target, transcribed) =>
      Seq(
        RecordDrawing(target, document, Axis[Channel]),
        RecordDrawing(transcribed, document, Axis[Channel]),
        RecordDrawing(transcribed, nothing, Axis[Channel])
      )
    val picture = Path.of(Runs.outputDir, s"d2g-${setup.corpus.name}-$size-$step.png")
    tiling.write(toPyTensor(stack(eachDrawing.flatten, Axis[Tile])), picture.toString, Rows, Across, eachDrawing.head.size)
    println(s"step $step | ${written.size} drawings written to $picture")

/** The picture: drawings across, rows down, each drawing shown as the three tiles above. */
private val Rows = 16
private val Across = 16

/** An empty canvas, as [[dataset.Outlines]] reads one. */
private val Blank = 255

/** How many drawings are transcribed together, as in the scoring: one traced computation. */
private val WrittenTogether = 32

/** Axis of the drawings laid out in one picture. */
private trait Tile derives Label

/** `drawing_tiles.py`, unpacked where the interpreter will find it. */
private lazy val tiling: py.Dynamic =
  Jax.np // DimWit configures the interpreter and `sys.path` before any Python object of ours.
  val directory = Files.createTempDirectory("d2g-drawings")
  py.module("sys").path.append(directory.toAbsolutePath.toString)
  val source = Option(classOf[Tile].getResourceAsStream("/python/drawing_tiles.py")).getOrElse(sys.error("drawing_tiles.py is not on the classpath"))
  try Files.copy(source, directory.resolve("drawing_tiles.py"), StandardCopyOption.REPLACE_EXISTING)
  finally source.close()
  py.module("drawing_tiles")
