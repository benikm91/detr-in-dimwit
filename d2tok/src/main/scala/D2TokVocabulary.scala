package d2tok

import dimwit.*

export documentEncoder.{Width, Height, Channel, Patch}

trait Node derives Label // A record node
trait Token derives Label // A position in the words a record is written in
trait Word derives Label // An entry of the vocabulary a record is written in
trait Part derives Label // A word of one node or relationship: its type, then what places or links it
trait Embedding derives Label // The (learned) latent vector spaces inside the model
trait Relationship derives Label // A record's relationships, which a sequence of words is predicted without

/** Coordinates are discrete here: a coordinate is the pixel it falls on. */
object Pixels:

  def of[T <: Tuple: Labels](coordinates: Tensor[T, Float32], canvas: Int): Tensor[T, Int32] =
    val pixel = (coordinates *! canvas.toFloat +! 0.5f).asInt(VType[Int32])
    minimum(pixel, Tensor.like(pixel).fill(canvas - 1))
