package dataset

import dimwit.*
import dimwit.Conversions.given

import scala.language.implicitConversions

/** Draws [[Detection]]s onto the drawing they belong to. */
object Outlines:

  /** Axis of [[shades]], one entry per [[ObjectClass]]. */
  private trait Shade derives Label

  /** Grey level of a blank canvas — the drawings are dark ink on white. */
  private val Blank = 255f

  /** Outline grey level per [[ObjectClass.id]]; an unused slot stays blank. */
  private def shades = Tensor1(Axis[Shade]).fromArray(Array(Blank, 110f, 170f))

  /** The image as the 8 bit grey levels [[plotwit]] plots. */
  def greyLevels[W: Label, H: Label, C: Label, V: IsFloating](image: Tensor3[W, H, C, V]): Tensor2[W, H, UInt8] =
    (image.squeeze(Axis[C]).asFloat32 *! 255f).asInt(VType[UInt8])

  /** The image with a one pixel outline around every object. */
  def apply[W: Label, H: Label, Slot: Label](image: Tensor2[W, H, UInt8], objects: Detection[Slot, Float32]): Tensor2[W, H, UInt8] =
    // Both are ink on a blank canvas, so the darker of the two wins per pixel.
    minimum(image, outlines(objects, image.shape))

  private def outlines[W: Label, H: Label, Slot: Label](
      objects: Detection[Slot, Float32],
      image: Shape2[W, H]
  ): Tensor2[W, H, UInt8] =
    val imageWidth = image(Axis[W])
    val imageHeight = image(Axis[H])
    val boxes = Shape3(
      Axis[W] -> imageWidth,
      Axis[H] -> imageHeight,
      Axis[Slot] -> objects.label.shape(Axis[Slot])
    )

    val x = coordinates(Axis[W], imageWidth).broadcastTo(boxes)
    val y = coordinates(Axis[H], imageHeight).broadcastTo(boxes)
    val centerX = (objects.box.centerX *! imageWidth.toFloat).broadcastTo(boxes)
    val centerY = (objects.box.centerY *! imageHeight.toFloat).broadcastTo(boxes)
    val halfWidth = (objects.box.width *! (imageWidth / 2f)).broadcastTo(boxes)
    val halfHeight = (objects.box.height *! (imageHeight / 2f)).broadcastTo(boxes)

    // Signed distance to the box border in pixels: negative inside, zero on it.
    val distance = maximum((x - centerX).abs - halfWidth, (y - centerY).abs - halfHeight)
    val onBorder = (distance +! 0.5f).abs <= Tensor(boxes).fill(0.5f)

    val shade = shades.take(Axis[Shade])(objects.label).broadcastTo(boxes)
    where(onBorder, shade, Tensor(boxes).fill(Blank)).min(Axis[Slot]).asInt(VType[UInt8])

  private def coordinates[L: Label](axis: Axis[L], size: Int): Tensor1[L, Float32] =
    Tensor1(axis).fromArray(Array.tabulate(size)(_.toFloat))
