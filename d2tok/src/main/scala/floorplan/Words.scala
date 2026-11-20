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
    val (words, said) = ofNodes(nodes)
    def inLine[W](of: Tensor2[Node, Part, W]) = of.flatten((Axis[Node], Axis[Part])).relabel(Axis[Node |*| Part] -> Axis[Token])
    inSequence(inLine(words), inLine(said), end)

  /** Every node's words along `Part` — its type, then its four coordinates — and which of them it
    * says: all of them, or none at a position the record does not reach.
    */
  def ofNodes(nodes: FloorplanNodes[Node]): (Tensor2[Node, Part, Int32], Tensor2[Node, Part, Bool]) =
    def pixel(coordinate: Tensor1[Node, Float32]) = Pixels.of(coordinate, canvas) +! classes.size
    val isNode = !(nodes.nodeClass elementEquals_! FloorplanClass.NoNode.id)
    zipvmap(Axis[Node])(nodes.nodeClass -! 1, pixel(nodes.startX), pixel(nodes.startY), pixel(nodes.endX), pixel(nodes.endY), isNode):
      case (typed, sx, sy, ex, ey, said) => (stack(Seq(typed, sx, sy, ex, ey), Axis[Part]), stack(Seq.fill(perNode)(said), Axis[Part]))

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
