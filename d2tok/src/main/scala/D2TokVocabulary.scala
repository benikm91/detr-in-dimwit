package d2tok

import dataset.NodeClass
import dataset.NodeClasses
import dataset.Point
import dataset.RecordNode
import dataset.RecordNodes
import dimwit.*

import scala.language.implicitConversions

export documentEncoder.{Width, Height, Channel, Patch}

trait Node derives Label // A record node
trait Token derives Label // A position in the words a record is written in
trait Word derives Label // An entry of the vocabulary a record is written in
trait Part derives Label // A word of one node: its type, then its coordinates
trait Embedding derives Label // The (learned) latent vector spaces inside the model
trait Relationship derives Label // A record's relationships, which a sequence of words is predicted without

/** The language a record is written in, on a canvas of `canvas` pixels a side.
  *
  * A node is a type word — its class, and whether it is construction geometry — followed by one
  * word per coordinate its class places, x before y, each the pixel it falls on. A record is the
  * words of its nodes one after the other, then a word that ends it; the first word is read after
  * a word that starts it.
  */
class Words(canvas: Int):

  private val classes = NodeClass.values.filter(_.isNode).toSeq
  private val coordinates = 2 * classes.size

  val end: Int = coordinates + canvas
  val start: Int = end + 1
  val size: Int = start + 1

  /** The most words a node writes: the type, and two coordinates for each point of the class that places most. */
  val perNode: Int = 1 + 2 * NodeClass.values.map(_.numPoints).max

  /** How many positions the words of a record of up to `nodes` nodes take, its end among them. */
  def tokens(nodes: Int): Int = nodes * perNode + 1

  /** The words of a record laid out along `Node`, and then its end, the positions after it ending it
    * again; and how many words it takes, its end among them.
    */
  def written(nodes: RecordNodes[Node], tokens: AxisExtent[Token]): (Tensor1[Token, Int32], Tensor0[Int32]) =
    val slots = nodes.nodeClass.shape(Axis[Node])
    require(tokens.size == this.tokens(slots), s"$slots nodes are written in ${this.tokens(slots)} words, not ${tokens.size}")
    def perClass(of: NodeClass => Int) = Tensor1(Axis[NodeClasses], VType[Int32]).fromArray(NodeClass.values.map(of)).slice(Axis[NodeClasses].at(nodes.nodeClass))
    val typed = perClass(nodeClass => 2 * classes.indexOf(nodeClass).max(0)) + nodes.construction
    val placed = perClass(nodeClass => if nodeClass.isNode then 1 + 2 * nodeClass.numPoints else 0)
    def pixel(coordinate: Tensor1[Node, Float32]) = Pixels.of(coordinate, canvas) +! coordinates
    val words = zipvmap(Axis[Node])(typed, pixel(nodes.startX), pixel(nodes.startY), pixel(nodes.endX), pixel(nodes.endY), pixel(nodes.midX), pixel(nodes.midY)):
      case (word, sx, sy, ex, ey, mx, my) => stack(Seq(word, sx, sy, ex, ey, mx, my), Axis[Part])
    val parts = Tensor1(Axis[Part]).fromRange(0 until perNode)
    val written = placed.vmap(Axis[Node])(count => parts <=! (count - 1))
    val flat = words.flatten((Axis[Node], Axis[Part]))
    val kept = written.flatten((Axis[Node], Axis[Part]))
    val at = Tensor1(Axis[Node |*| Part]).fromRange(0 until slots * perNode)
    val inOrder = flat.slice(Axis[Node |*| Part].at(where(kept, at, at +! slots * perNode).argsort(Axis[Node |*| Part])))
    val count = kept.asInt(VType[Int32]).sum
    val ended = concatenate(inOrder, Tensor0(end).prependAxis(Axis[Node |*| Part]), Axis[Node |*| Part]).relabel(Axis[Node |*| Part] -> Axis[Token])
    val positions = Tensor1(Axis[Token]).fromRange(0 until tokens.size)
    (where(positions <=! (count - 1), ended, Tensor.like(ended).fill(end)), count + 1)

  /** The nodes `words` write, up to their end, or to the first word that does not fit where it stands. */
  def read(words: Seq[Int]): Seq[RecordNode] =
    def isCoordinate(word: Int) = word >= coordinates && word < end
    Iterator.unfold(words):
      case typed +: rest if typed < coordinates =>
        val (nodeClass, isConstruction) = (classes(typed / 2), typed % 2 == 1)
        val placed = rest.take(2 * nodeClass.numPoints)
        Option.when(placed.size == 2 * nodeClass.numPoints && placed.forall(isCoordinate)):
          val points = placed.grouped(2).map(xy => Point((xy(0) - coordinates).toFloat / canvas, (xy(1) - coordinates).toFloat / canvas)).toSeq
          (RecordNode(nodeClass, isConstruction, points), rest.drop(placed.size))
      case _ => None
    .toSeq

/** Coordinates are discrete here: a coordinate is the pixel it falls on. */
object Pixels:

  def of[T <: Tuple: Labels](coordinates: Tensor[T, Float32], canvas: Int): Tensor[T, Int32] =
    val pixel = (coordinates *! canvas.toFloat +! 0.5f).asInt(VType[Int32])
    minimum(pixel, Tensor.like(pixel).fill(canvas - 1))
