package detr

import detr.model.*
import detr.train.*
import detr.eval.*
import detr.config.*
import dimwit.*

export documentEncoder.{Width, Height, Channel, Patch}

/** Axis of DETR's object queries, each of which answers with one node of the record or with none. */
trait Query derives Label

/** Axis of the nodes of a record, which is what the queries are matched against. */
trait Node derives Label

/** Axis of the relationships a record holds between its nodes, which a detector does not predict
  * but the records it is trained on still carry.
  */
trait Relationship derives Label

/** A pixel coordinate in the image. */
trait Pixel derives Label

/** Coordinates are discrete here: a coordinate is the pixel it falls on. */
object Pixels:

  def of[T <: Tuple: Labels](coordinates: Tensor[T, Float32], canvas: Int): Tensor[T, Int32] =
    val pixel = (coordinates *! canvas.toFloat +! 0.5f).asInt(VType[Int32])
    minimum(pixel, Tensor.like(pixel).fill(canvas - 1))

  def coordinates[T <: Tuple: Labels](pixels: Tensor[T, Int32], canvas: Int): Tensor[T, Float32] =
    pixels.asFloat(VType[Float32]) /! canvas.toFloat
