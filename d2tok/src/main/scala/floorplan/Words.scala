package d2tok.floorplan

import d2tok.*
import dataset.Point
import dimwit.*
import syntheticFloorplansDataset.FloorplanClass
import syntheticFloorplansDataset.FloorplanNode
import syntheticFloorplansDataset.FloorplanNodes

import scala.language.implicitConversions

/** The language a floor plan's record is written in, on a canvas of `canvas` pixels a side.
  *
  * A node is a type word — its class — followed by four coordinate words, the pixels its start and
  * its end fall on, x before y. A record is the words of its nodes one after the other, then a word
  * that ends it; the first word is read after a word that starts it.
  */
class Words(canvas: Int):

  private val classes = FloorplanClass.values.filter(_.isNode).toSeq

  val end: Int = classes.size + canvas
  val start: Int = end + 1
  val size: Int = start + 1

  /** The words every node writes: the type, and the two coordinates of either end. */
  val perNode: Int = 5

  /** How many positions the words of a record of up to `nodes` nodes take, its end among them. */
  def tokens(nodes: Int): Int = nodes * perNode + 1

  /** The words of a record laid out along `Node`, its nodes first, and then its end, the positions
    * after it ending it again; and how many words it takes, its end among them.
    */
  def written(nodes: FloorplanNodes[Node], tokens: AxisExtent[Token]): (Tensor1[Token, Int32], Tensor0[Int32]) =
    val slots = nodes.nodeClass.shape(Axis[Node])
    require(tokens.size == this.tokens(slots), s"$slots nodes are written in ${this.tokens(slots)} words, not ${tokens.size}")
    def pixel(coordinate: Tensor1[Node, Float32]) = Pixels.of(coordinate, canvas) +! classes.size
    val typed = nodes.nodeClass -! 1
    val words = zipvmap(Axis[Node])(typed, pixel(nodes.startX), pixel(nodes.startY), pixel(nodes.endX), pixel(nodes.endY)):
      case (word, sx, sy, ex, ey) => stack(Seq(word, sx, sy, ex, ey), Axis[Part])
    val flat = words.flatten((Axis[Node], Axis[Part]))
    val ended = concatenate(flat, Tensor0(end).prependAxis(Axis[Node |*| Part]), Axis[Node |*| Part]).relabel(Axis[Node |*| Part] -> Axis[Token])
    val held = (!(nodes.nodeClass elementEquals_! FloorplanClass.NoNode.id)).asInt(VType[Int32]).sum
    val count = held * perNode
    val positions = Tensor1(Axis[Token]).fromRange(0 until tokens.size)
    (where(positions <=! (count - 1), ended, Tensor.like(ended).fill(end)), count + 1)

  /** The nodes `words` write, up to their end, or to the first word that does not fit where it stands. */
  def read(words: Seq[Int]): Seq[FloorplanNode] =
    def isCoordinate(word: Int) = word >= classes.size && word < end
    def at(x: Int, y: Int) = Point((x - classes.size).toFloat / canvas, (y - classes.size).toFloat / canvas)
    Iterator.unfold(words):
      case typed +: rest if typed < classes.size =>
        val placed = rest.take(4)
        Option.when(placed.size == 4 && placed.forall(isCoordinate)):
          (FloorplanNode(classes(typed), at(placed(0), placed(1)), at(placed(2), placed(3))), rest.drop(4))
      case _ => None
    .toSeq
