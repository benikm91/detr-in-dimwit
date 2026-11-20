package d2tok

import dimwit.*

import scala.language.implicitConversions

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

/** The words `said` of `words`, one after the other and then `end`, the positions after it ending
  * the record again; and how many words that is, its end among them. One position longer than
  * `words`, for the end.
  */
def inSequence(words: Tensor1[Token, Int32], said: Tensor1[Token, Bool], end: Int): (Tensor1[Token, Int32], Tensor0[Int32]) =
  val slots = words.shape(Axis[Token])
  val at = Tensor1(Axis[Token]).fromRange(0 until slots)
  val inOrder = words.slice(Axis[Token].at(where(said, at, at +! slots).argsort(Axis[Token])))
  val count = said.asInt(VType[Int32]).sum
  val ended = concatenate(inOrder, Tensor0(end).prependAxis(Axis[Token]), Axis[Token])
  val positions = Tensor1(Axis[Token]).fromRange(0 until slots + 1)
  (where(positions <=! (count - 1), ended, Tensor.like(ended).fill(end)), count + 1)
