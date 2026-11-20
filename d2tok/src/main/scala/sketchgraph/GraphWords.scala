package d2tok.sketchgraph

import d2tok.*
import d2tok.sketch.Words
import dataset.EdgeClass
import dataset.NodeClass
import dataset.Record
import dataset.RecordEdge
import dataset.RecordGraph
import dimwit.*

import scala.language.implicitConversions

/** The language a record and its relationships are written in, as one sequence, for records of up
  * to `nodes` nodes on a canvas of `canvas` pixels a side.
  *
  * The nodes come first, each its number and then its words in [[Words]], numbered 0, 1, 2, … as
  * they are written. Each relationship follows as a type word and the numbers of the two nodes it
  * links. Then a word ends the record. The words of [[Words]] keep their numbers, so its start and
  * end are this language's too.
  */
class GraphWords(canvas: Int, nodes: Int):

  val nodeWords: Words = Words(canvas)
  export nodeWords.{end, start}

  private val relationships = EdgeClass.values.filter(_.isEdge).toSeq
  private val firstType = nodeWords.size
  private val firstNumber = firstType + relationships.size

  val size: Int = firstNumber + nodes

  /** The most words a node writes: its number, then its words. */
  val perNode: Int = 1 + nodeWords.perNode

  /** The words every relationship writes: its type, and the numbers of the two nodes it links. */
  val perRelationship: Int = 3

  /** How many positions the words of up to `nodes` nodes and `relationships` relationships take,
    * the end among them.
    */
  def tokens(nodes: Int, relationships: Int): Int = nodes * perNode + relationships * perRelationship + 1

  /** The words of a record laid out along `Node` and `Relationship`, and then its end, the positions
    * after it ending it again; and how many words it takes, its end among them.
    */
  def written(record: Record[Node, Relationship], tokens: AxisExtent[Token]): (Tensor1[Token, Int32], Tensor0[Int32]) =
    val (nodeSlots, relationshipSlots) = (record.nodeClass.shape(Axis[Node]), record.edgeClass.shape(Axis[Relationship]))
    require(tokens.size == this.tokens(nodeSlots, relationshipSlots), s"$nodeSlots nodes and $relationshipSlots relationships are written in ${this.tokens(nodeSlots, relationshipSlots)} words, not ${tokens.size}")

    val (ofNodes, saidOfNodes) = nodeWords.ofNodes(record.nodes)
    val number = Tensor1(Axis[Node]).fromRange(0 until nodeSlots) +! firstNumber
    val isNode = !(record.nodeClass elementEquals_! NodeClass.NoNode.id)
    val numbered = concatenate(number.appendAxis(Axis[Part]), ofNodes, Axis[Part])
    val saidNumbered = concatenate(isNode.appendAxis(Axis[Part]), saidOfNodes, Axis[Part])

    def numberOf(node: Tensor1[Relationship, Int32]) = node +! firstNumber
    val isRelationship = !(record.edgeClass elementEquals_! EdgeClass.NoEdge.id)
    val (ofRelationships, saidOfRelationships) =
      zipvmap(Axis[Relationship])(record.edgeClass +! (firstType - 1), numberOf(record.subject), numberOf(record.obj), isRelationship):
        case (typed, subject, obj, said) => (stack(Seq(typed, subject, obj), Axis[Part]), stack(Seq(said, said, said), Axis[Part]))

    def inLine[W](byNode: Tensor2[Node, Part, W], byRelationship: Tensor2[Relationship, Part, W]) = concatenate(
      byNode.flatten((Axis[Node], Axis[Part])).relabel(Axis[Node |*| Part] -> Axis[Token]),
      byRelationship.flatten((Axis[Relationship], Axis[Part])).relabel(Axis[Relationship |*| Part] -> Axis[Token]),
      Axis[Token]
    )
    inSequence(inLine(numbered, ofRelationships), inLine(saidNumbered, saidOfRelationships), end)

  /** The record `words` write, up to their end, or to the first word that does not fit where it
    * stands: a node whose number is not the next one does not, and neither does a relationship
    * naming a node that was not written.
    */
  def read(words: Seq[Int]): RecordGraph =
    val drawn = Iterator.unfold((words, 0)):
      case (number +: rest, next) if number == firstNumber + next =>
        nodeWords.read(rest).headOption.map(node => (node, (rest.drop(1 + 2 * node.nodeClass.numPoints), next + 1)))
      case _ => None
    .toSeq
    val afterNodes = words.drop(drawn.map(node => 2 + 2 * node.nodeClass.numPoints).sum)
    def namesNode(word: Int) = word >= firstNumber && word - firstNumber < drawn.size
    val related = Iterator.unfold(afterNodes):
      case typed +: subject +: obj +: rest if typed >= firstType && typed < firstNumber && namesNode(subject) && namesNode(obj) =>
        Some((RecordEdge(relationships(typed - firstType), subject - firstNumber, obj - firstNumber), rest))
      case _ => None
    .toSeq
    RecordGraph(drawn, related)
