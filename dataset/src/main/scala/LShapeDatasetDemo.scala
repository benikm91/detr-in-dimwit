package dataset

import dataset.LShapeDetectionDataset.Split
import dimwit.*
import plotwit.*
import viz.PlotTargets.websocket

/** The axes the demo labels the loaded tensors with. */
private trait Width derives Label
private trait Height derives Label
private trait Channel derives Label
private trait BoundingBox derives Label

/** Shows what [[LShapeDetectionDataset]] hands out: `sbt "dataset/runMain dataset.lShapeDemo"`.
  *
  * Reads the validation split (67 MB, downloaded into the HuggingFace cache on first run)
  * and plots the first few drawings next to the same drawings with their object boxes drawn
  * on top.
  */
@main
def lShapeDemo(): Unit =
  dimwit.initialize()

  val data = LShapeDetectionDataset.open(Axis[Width], Axis[Height], Axis[Channel], Axis[BoundingBox])(Split.Validation)
  println(data)
  println(s"most objects in a sample: ${data.maxObjects}")

  val rows = data.samples().take(3).zipWithIndex.map: (sample, index) =>
    val drawing = Outlines.greyLevels(sample.image)
    println(s"sample $index: ${sample.objects.label}")
    Seq(
      plots.imagePlot(drawing, _.title := s"sample $index"),
      plots.imagePlot(Outlines(drawing, sample.objects), _.title := s"sample $index — objects")
    )

  display(grid(rows.toSeq))
