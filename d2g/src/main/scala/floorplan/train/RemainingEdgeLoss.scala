package d2g.floorplan.train

import d2g.*
import d2g.floorplan.model.*
import d2s.model.*
import d2s.train.*
import EdgeHead.EdgeLogits
import dimwit.*
import syntheticFloorplansDataset.FloorplanEdges
import syntheticFloorplansDataset.FloorplanRelationship
import syntheticFloorplansDataset.FloorplanRelationships

import scala.language.implicitConversions

/** What a floor plan's relationships cost, predicted the way [[d2s.floorplan.train.RemainingNodeLoss]]
  * predicts its nodes — a relationship carries the nodes it relates where a node carries its points.
  *
  * @param linkSmoothing How much of each node a relationship relates is spread evenly over every
  *                      node instead.
  */
class RemainingEdgeLoss[V: IsFloating](vtype: VType[V], linkSmoothing: Float)
    extends ((D2G.EdgeQueryLogits[V], FloorplanEdges[Edge]) => Tensor0[V]):

  /** Axis of the record's relationships seen as candidates to answer with rather than as positions. */
  private type Candidate = Prime[Edge]

  override def apply(answered: D2G.EdgeQueryLogits[V], target: FloorplanEdges[Edge]): Tensor0[V] =
    require(answered.relationship.shape(Axis[PoolQuery]) == 2, "the loss pairs two answers")
    val (a, b) = (answered.at(0), answered.at(1))
    val edges = target.relationship.shape.extent(Axis[Edge])
    val pairs = Shape2(edges, Axis[Candidate] -> edges.size)
    val holdsEdge = FloorplanRelationship.indicator(vtype)(_.isRelationship).slice(Axis[FloorplanRelationships].at(target.relationship))
    val taken = holdsEdge.sum
    val candidates = triu(Tensor(pairs, vtype).fill(1f)) *! holdsEdge.relabelTo(Axis[Candidate])

    val asked = candidates.max(Axis[Candidate])
    val guessed = distinctly(dissimilarity(a, target), dissimilarity(b, target), candidates)
    val ended = isAt(taken, edges, vtype)
    val stops = costOfClass(a.relationship, FloorplanRelationship.NoRelationship.id) + costOfClass(b.relationship, FloorplanRelationship.NoRelationship.id)
    ((guessed * asked).sum + (stops * ended).sum) / ((taken + 1f) * 2f)

  /** What every position's scores would cost against every relationship of the record: what it is,
    * and the two nodes it relates. Only relationships are ever candidates, so both ends always count.
    */
  private def dissimilarity(logits: EdgeLogits[V], target: FloorplanEdges[Edge]): Tensor2[Edge, Candidate, V] =
    def named(scores: Tensor2[Edge, LinkedNode, V], end: Tensor1[Edge, Int32]) =
      val spread = logNormalizer(scores) - scores.mean(Axis[LinkedNode])
      spread *! linkSmoothing +! costOfValue(scores, end.relabelTo(Axis[Candidate])) *! (1f - linkSmoothing)
    costOfValue(logits.relationship, target.relationship.relabelTo(Axis[Candidate])) +
      named(logits.subject, target.subject) +
      named(logits.obj, target.obj)
