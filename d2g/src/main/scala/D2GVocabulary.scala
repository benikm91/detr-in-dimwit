package d2g

import d2g.model.*
import d2g.train.*
import d2g.eval.*
import d2g.config.*
import dimwit.*
import dimwit.Conversions.given

import scala.language.implicitConversions

export documentEncoder.{Width, Height, Channel, Patch}

// Nodes and their properties
trait Node derives Label // A record node
trait Pixel derives Label // A pixel coordinate in the image

// Edges and their properties
trait Edge derives Label // A record edge
trait LinkedNode derives Label // The node a relationship links to

trait Embedding derives Label // The (learned) latent vector spaces inside the model

/** Coordinates are discrete here: a coordinate is the pixel it falls on. */
object Pixels:

  def of[T <: Tuple: Labels](coordinates: Tensor[T, Float32], canvas: Int): Tensor[T, Int32] =
    val pixel = (coordinates *! canvas.toFloat +! 0.5f).asInt(VType[Int32])
    minimum(pixel, Tensor.like(pixel).fill(canvas - 1))

  def coordinates[T <: Tuple: Labels](pixels: Tensor[T, Int32], canvas: Int): Tensor[T, Float32] =
    pixels.asFloat(VType[Float32]) /! canvas.toFloat
