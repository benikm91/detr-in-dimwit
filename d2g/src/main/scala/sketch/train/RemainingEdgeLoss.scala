package d2g.sketch.train

import d2g.*
import d2g.sketch.RelationshipOrder
import d2g.sketch.model.*
import d2s.model.*
import d2s.train.*
import dataset.EdgeClass
import dataset.EdgeClasses
import dataset.Record
import dataset.RecordEdges
import EdgeHead.EdgeLogits
import dimwit.*

import scala.language.implicitConversions

/** What a drawing's relationships cost, predicted the way [[d2s.sketch.train.RemainingNodeLoss]]
  * predicts its nodes — a relationship carries the nodes it relates where a node carries its points,
  * and is answered among the remaining ones of its own class where they are `written` class by class.
  */
class RemainingEdgeLoss[V: IsFloating](vtype: VType[V], written: RelationshipOrder):

  /** Axis of the record's relationships seen as candidates to answer with rather than as positions. */
  private type Candidate = Prime[Edge]

  def apply(answered: D2G.EdgeQueryLogits[V], record: Record[Node, Edge]): Tensor0[V] =
    require(answered.edgeClass.shape(Axis[PoolQuery]) == 2, "the loss pairs two answers")
    val (a, b) = (answered.at(0), answered.at(1))
    val target = record.edges
    val edges = target.edgeClass.shape.extent(Axis[Edge])
    val pairs = Shape2(edges, Axis[Candidate] -> edges.size)
    val holdsEdge = EdgeClass.indicator(vtype)(_.isEdge).slice(Axis[EdgeClasses].at(target.edgeClass))
    val taken = holdsEdge.sum
    val remaining = triu(Tensor(pairs, vtype).fill(1f)) *! holdsEdge.relabelTo(Axis[Candidate])
    val candidates = written match
      case RelationshipOrder.Free   => remaining
      case RelationshipOrder.ByKind =>
        val kind = written.classRank.slice(Axis[EdgeClasses].at(target.edgeClass))
        val sameKind = kind.vmap(Axis[Edge])(own => kind.relabelTo(Axis[Candidate]) elementEquals_! own)
        remaining * sameKind.asFloat(vtype)

    val asked = candidates.max(Axis[Candidate])
    val guessed = distinctly(dissimilarity(a, target), dissimilarity(b, target), candidates)
    val ended = isAt(taken, edges, vtype)
    val stops = costOfClass(a.edgeClass, EdgeClass.NoEdge.id) + costOfClass(b.edgeClass, EdgeClass.NoEdge.id)
    ((guessed * asked).sum + (stops * ended).sum) / ((taken + 1f) * 2f)

  /** What every position's scores would cost against every relationship of the record: what it is,
    * and the two nodes it relates. Only relationships are ever candidates, so both ends always count.
    */
  private def dissimilarity(logits: EdgeLogits[V], target: RecordEdges[Edge]): Tensor2[Edge, Candidate, V] =
    def named(scores: Tensor2[Edge, Node, V], end: Tensor1[Edge, Int32]) =
      costOfValue(scores, end.relabelTo(Axis[Candidate]))
    costOfValue(logits.edgeClass, target.edgeClass.relabelTo(Axis[Candidate])) +
      named(logits.subject, target.subject) +
      named(logits.obj, target.obj)
