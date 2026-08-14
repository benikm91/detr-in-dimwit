package dataset

import dataset.LShapeDetectionDataset.Split
import dimwit.*
import dimwit.Conversions.given
import plotwit.*
import viz.PlotTargets.websocket

import scala.language.implicitConversions

/** The axes the demo labels the loaded tensors with. */
private trait Width derives Label
private trait Height derives Label
private trait Channel derives Label
private trait BoundingBox derives Label

/** Axis of [[outlineShades]], one entry per [[ObjectClass]]. */
private trait Shade derives Label

/** Grey level of a blank canvas — the drawings are dark ink on white. */
private val Blank = 255f

/** Outline grey level per `ObjectClass.id`; an unused query slot stays blank. */
private def outlineShades = Tensor1(Axis[Shade]).fromArray(Array(Blank, 110f, 170f))

/** Shows what [[LShapeDetectionDataset]] hands out: `sbt "dataset/runMain dataset.lShapeDemo"`.
  *
  * Reads the validation split (67 MB, downloaded into the HuggingFace cache on first run)
  * and plots the first few drawings next to the same drawings with their object boxes
  * drawn on top.
  */
@main
def lShapeDemo(): Unit =
  dimwit.initialize()

  val data = LShapeDetectionDataset.open(Axis[Width], Axis[Height], Axis[Channel], Axis[BoundingBox])(Split.Validation)
  println(data)
  println(s"most objects in a sample: ${data.maxObjects}")

  val rows = data.samples().take(3).zipWithIndex.map: (sample, index) =>
    val drawing = greyLevels(sample.image)
    println(s"sample $index: ${sample.objects.label}")
    Seq(
      plots.imagePlot(drawing, _.title := s"sample $index"),
      plots.imagePlot(minimum(drawing, outlines(sample.objects, drawing.shape)), _.title := s"sample $index — objects")
    )

  display(grid(rows.toSeq))

private def greyLevels(image: Tensor3[Width, Height, Channel, Float32]): Tensor2[Width, Height, UInt8] =
  (image.squeeze(Axis[Channel]) *! 255f).asInt(VType[UInt8])

/** A blank canvas carrying a one pixel outline around every detected object. */
private def outlines(objects: Detection[BoundingBox], image: Shape2[Width, Height]): Tensor2[Width, Height, UInt8] =
  val imageWidth = image(Axis[Width])
  val imageHeight = image(Axis[Height])
  val boxes = Shape3(
    Axis[Width] -> imageWidth,
    Axis[Height] -> imageHeight,
    Axis[BoundingBox] -> objects.label.shape(Axis[BoundingBox])
  )

  val x = coordinates(Axis[Width], imageWidth).broadcastTo(boxes)
  val y = coordinates(Axis[Height], imageHeight).broadcastTo(boxes)
  val centerX = (objects.centerX *! imageWidth.toFloat).broadcastTo(boxes)
  val centerY = (objects.centerY *! imageHeight.toFloat).broadcastTo(boxes)
  val halfWidth = (objects.width *! (imageWidth / 2f)).broadcastTo(boxes)
  val halfHeight = (objects.height *! (imageHeight / 2f)).broadcastTo(boxes)

  // Signed distance to the box border in pixels: negative inside, zero on it.
  val distance = maximum((x - centerX).abs - halfWidth, (y - centerY).abs - halfHeight)
  val onBorder = (distance +! 0.5f).abs <= Tensor(boxes).fill(0.5f)

  val shade = outlineShades.take(Axis[Shade])(objects.label).broadcastTo(boxes)
  where(onBorder, shade, Tensor(boxes).fill(Blank)).min(Axis[BoundingBox]).asInt(VType[UInt8])

private def coordinates[L: Label](axis: Axis[L], size: Int): Tensor1[L, Float32] =
  Tensor1(axis).fromArray(Array.tabulate(size)(_.toFloat))
