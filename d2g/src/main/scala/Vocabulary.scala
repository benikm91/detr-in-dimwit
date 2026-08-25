import dimwit.*

trait Width derives Label
trait Height derives Label
trait Channel derives Label

/** Axis of the record's drawn nodes, which is also what a relationship links them by. */
trait Node derives Label

/** Axis of the record's relationships. */
trait Edge derives Label

/** Axis of the document flattened into the encoder's sequence of patches. */
type Patch = Width |*| Height

/** Axis over the pixel a coordinate is predicted as. Coordinates are discrete here, which is all
  * the precision a drawing has anyway.
  */
trait Pixel derives Label

/** Axis over the node a link is predicted to name. */
trait LinkedNode derives Label

/** Axis of the space the decoder works in. */
trait Embedding derives Label

/** Axis of the pieces a node embedding is put together from: its class and the points it is
  * placed by.
  */
trait NodePart derives Label

/** Axis of the pieces a relationship embedding is put together from: its class and the nodes it
  * links.
  */
trait EdgePart derives Label

/** Axis of the space one such piece is embedded in. */
trait PartEmbedding derives Label

/** Coordinates are discrete here: a coordinate is the pixel it falls on. */
object Pixels:

  def of[T <: Tuple: Labels](coordinates: Tensor[T, Float32], canvas: Int): Tensor[T, Int32] =
    val pixel = (coordinates * Tensor.like(coordinates).fill(canvas.toFloat) + Tensor.like(coordinates).fill(0.5f)).asInt(VType[Int32])
    minimum(pixel, Tensor.like(pixel).fill(canvas - 1))

  def coordinates[T <: Tuple: Labels](pixels: Tensor[T, Int32], canvas: Int): Tensor[T, Float32] =
    pixels.asFloat(VType[Float32]) / Tensor.like(pixels).fill(canvas).asFloat(VType[Float32])
