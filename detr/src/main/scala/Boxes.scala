import dimwit.*
import dimwit.Conversions.given

import scala.language.implicitConversions

/** Axis aligned boxes in `(centerX, centerY, width, height)` form, normalized to `[0, 1]`
  * of the image.
  */
case class Box[T <: Tuple, V](
    centerX: Tensor[T, V],
    centerY: Tensor[T, V],
    width: Tensor[T, V],
    height: Tensor[T, V]
):
  def map[R <: Tuple](f: Tensor[T, V] => Tensor[R, V]): Box[R, V] =
    Box(f(centerX), f(centerY), f(width), f(height))

object Box:

  def l1[T <: Tuple: Labels, V: IsFloating](predicted: Box[T, V], target: Box[T, V]): Tensor[T, V] =
    (predicted.centerX - target.centerX).abs +
      (predicted.centerY - target.centerY).abs +
      (predicted.width - target.width).abs +
      (predicted.height - target.height).abs

  /** Generalized intersection over union, as in [[https://giou.stanford.edu]]. */
  def giou[T <: Tuple: Labels, V: IsFloating](predicted: Box[T, V], target: Box[T, V]): Tensor[T, V] =
    val a = corners(predicted)
    val b = corners(target)
    val overlap = extent(maximum(a.left, b.left), minimum(a.right, b.right)) *
      extent(maximum(a.top, b.top), minimum(a.bottom, b.bottom))
    val union = area(predicted) + area(target) - overlap + ε(predicted.centerX)
    val hull = extent(minimum(a.left, b.left), maximum(a.right, b.right)) *
      extent(minimum(a.top, b.top), maximum(a.bottom, b.bottom)) + ε(predicted.centerX)
    overlap / union - (hull - union) / hull

  private case class Corners[T <: Tuple, V](
      left: Tensor[T, V],
      right: Tensor[T, V],
      top: Tensor[T, V],
      bottom: Tensor[T, V]
  )

  private def corners[T <: Tuple: Labels, V: IsFloating](box: Box[T, V]): Corners[T, V] =
    Corners(
      left = box.centerX - box.width *! 0.5f,
      right = box.centerX + box.width *! 0.5f,
      top = box.centerY - box.height *! 0.5f,
      bottom = box.centerY + box.height *! 0.5f
    )

  private def area[T <: Tuple: Labels, V: IsFloating](box: Box[T, V]): Tensor[T, V] =
    box.width * box.height

  private def extent[T <: Tuple: Labels, V: IsFloating](from: Tensor[T, V], to: Tensor[T, V]): Tensor[T, V] =
    maximum(to - from, Tensor.like(from).fill(0f))

  private def ε[T <: Tuple: Labels, V: IsFloating](like: Tensor[T, V]): Tensor[T, V] =
    Tensor.like(like).fill(1e-6f)
