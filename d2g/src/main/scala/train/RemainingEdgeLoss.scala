package d2g.train

import d2g.*
import d2g.model.*
import d2s.model.*
import d2s.train.*
import dataset.EdgeClass
import dataset.EdgeClasses
import dataset.RecordEdges
import EdgeHead.EdgeLogits
import dimwit.*

import scala.language.implicitConversions

/** The same over the relationships of a record, which are predicted the same way — a relationship
  * carries the nodes it links where a node carries its points.
  */
class RemainingEdgeLoss[V: IsFloating](vtype: VType[V])
    extends ((D2G.EdgeQueryLogits[V], RecordEdges[Edge]) => Tensor0[V]):

  /** Axis of the record's relationships seen as candidates to answer with rather than as positions. */
  private type Candidate = Prime[Edge]

  override def apply(answered: D2G.EdgeQueryLogits[V], target: RecordEdges[Edge]): Tensor0[V] =
    require(answered.edgeClass.shape(Axis[PoolQuery]) == 2, "the loss pairs two answers")
    val (a, b) = (answered.at(0), answered.at(1))
    val edges = target.edgeClass.shape.extent(Axis[Edge])
    val pairs = Shape2(edges, Axis[Candidate] -> edges.size)
    val holdsEdge = EdgeClass.indicator(vtype)(_.isEdge).take(Axis[EdgeClasses])(target.edgeClass)
    val taken = holdsEdge.sum
    val candidates = triu(Tensor(pairs, vtype).fill(1f)) *! holdsEdge.relabelTo(Axis[Candidate])

    val asked = candidates.max(Axis[Candidate])
    val guessed = distinctly(dissimilarity(a, target), dissimilarity(b, target), candidates)
    val ended = isAt(taken, edges, vtype)
    val stops = costOfClass(a.edgeClass, EdgeClass.NoEdge.id) + costOfClass(b.edgeClass, EdgeClass.NoEdge.id)
    ((guessed * asked).sum + (stops * ended).sum) / ((taken + 1f) * 2f)

  /** The same for a relationship, which carries the two nodes it relates where a node carries the
    * points it is placed by. Only relationships are ever candidates, so both ends always count.
    */
  private def dissimilarity(logits: EdgeLogits[V], target: RecordEdges[Edge]): Tensor2[Edge, Candidate, V] =
    def named(scores: Tensor2[Edge, LinkedNode, V], end: Tensor1[Edge, Int32]) =
      costOfValue(scores, end.relabelTo(Axis[Candidate]))
    costOfValue(logits.edgeClass, target.edgeClass.relabelTo(Axis[Candidate])) +
      named(logits.subject, target.subject) +
      named(logits.obj, target.obj)
