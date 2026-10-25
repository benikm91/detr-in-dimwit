package d2tok.floorplangraph

import d2tok.*
import dimwit.*
import syntheticFloorplansDataset.FloorplanEdge
import syntheticFloorplansDataset.FloorplanEdges
import syntheticFloorplansDataset.FloorplanRelationship

import scala.language.implicitConversions

trait RelationshipToken derives Label // A position in the words a record's relationships are written in

/** The language a floor plan's relationships are written in, after its nodes, for records of up to
  * `nodes` nodes.
  *
  * A relationship is a type word — joined or in — followed by the numbers of the two nodes it
  * links, a node numbered by its place among the nodes as they were written. A record's
  * relationships are written one after the other, then a word that ends them; the first word is
  * read after a word that starts them.
  */
class RelationshipWords(nodes: Int):

  private val relationships = FloorplanRelationship.values.filter(_.isRelationship).toSeq

  val end: Int = relationships.size + nodes
  val start: Int = end + 1
  val size: Int = start + 1

  /** The words that name a node, by its number. */
  val numbers: Range = relationships.size until end

  /** The words every relationship writes: its type, and the numbers of the two nodes it links. */
  val perRelationship: Int = 3

  /** How many positions the words of up to `relationships` relationships take, their end among them. */
  def tokens(relationships: Int): Int = relationships * perRelationship + 1

  /** The words of the relationships laid out along `Relationship`, and then their end, the positions
    * after it ending them again; and how many words they take, their end among them.
    */
  def written(edges: FloorplanEdges[Relationship], tokens: AxisExtent[RelationshipToken]): (Tensor1[RelationshipToken, Int32], Tensor0[Int32]) =
    val slots = edges.relationship.shape(Axis[Relationship])
    require(tokens.size == this.tokens(slots), s"$slots relationships are written in ${this.tokens(slots)} words, not ${tokens.size}")
    def number(node: Tensor1[Relationship, Int32]) = node +! relationships.size
    val typed = edges.relationship -! 1
    val words = zipvmap(Axis[Relationship])(typed, number(edges.subject), number(edges.obj)):
      case (word, subject, obj) => stack(Seq(word, subject, obj), Axis[Part])
    val flat = words.flatten((Axis[Relationship], Axis[Part]))
    val ended = concatenate(flat, Tensor0(end).prependAxis(Axis[Relationship |*| Part]), Axis[Relationship |*| Part]).relabel(Axis[Relationship |*| Part] -> Axis[RelationshipToken])
    val held = (!(edges.relationship elementEquals_! FloorplanRelationship.NoRelationship.id)).asInt(VType[Int32]).sum
    val count = held * perRelationship
    val positions = Tensor1(Axis[RelationshipToken]).fromRange(0 until tokens.size)
    (where(positions <=! (count - 1), ended, Tensor.like(ended).fill(end)), count + 1)

  /** The relationships `words` write, up to their end, or to the first word that does not fit where it stands. */
  def read(words: Seq[Int]): Seq[FloorplanEdge] =
    Iterator.unfold(words):
      case typed +: subject +: obj +: rest if typed < relationships.size && numbers.contains(subject) && numbers.contains(obj) =>
        Some((FloorplanEdge(relationships(typed), subject - relationships.size, obj - relationships.size), rest))
      case _ => None
    .toSeq
