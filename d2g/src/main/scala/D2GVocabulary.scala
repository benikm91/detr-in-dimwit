package d2g

import d2g.model.*
import d2g.train.*
import d2g.eval.*
import d2g.config.*
import dimwit.*

trait Width derives Label
trait Height derives Label
trait Channel derives Label

trait Node derives Label // A record node
trait Edge derives Label // A record edge

type Patch = Width |*| Height // 1D patch sequence represent 2D image grid

trait Pixel derives Label // A pixel coordinate in the image

trait LinkedNode derives Label // The node a relationship links to

trait Embedding derives Label // The (learned) latent vector spaces inside the model

trait NodePart derives Label // The parts a node embedding is composed of

trait EdgePart derives Label // The parts a edge embedding is composed of

trait PartEmbedding derives Label // An embedding of a [[NodePart]] or a [[EdgePart]]

/** Coordinates are discrete here: a coordinate is the pixel it falls on. */
object Pixels:

  def of[T <: Tuple: Labels](coordinates: Tensor[T, Float32], canvas: Int): Tensor[T, Int32] =
    val pixel = (coordinates * Tensor.like(coordinates).fill(canvas.toFloat) + Tensor.like(coordinates).fill(0.5f)).asInt(VType[Int32])
    minimum(pixel, Tensor.like(pixel).fill(canvas - 1))

  def coordinates[T <: Tuple: Labels](pixels: Tensor[T, Int32], canvas: Int): Tensor[T, Float32] =
    pixels.asFloat(VType[Float32]) / Tensor.like(pixels).fill(canvas).asFloat(VType[Float32])
