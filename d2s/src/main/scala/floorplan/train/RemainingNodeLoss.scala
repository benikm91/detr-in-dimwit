package d2s.floorplan.train

import d2s.*
import d2s.model.*
import d2s.train.*
import d2s.floorplan.NodeOrder
import d2s.floorplan.model.*
import NodeHead.NodeLogits
import dimwit.*
import syntheticFloorplansDataset.FloorplanClass
import syntheticFloorplansDataset.FloorplanClasses
import syntheticFloorplansDataset.FloorplanNodes

import scala.language.implicitConversions

/** What a floor plan's nodes cost the model that writes them down, in the order they are `written`
  * in, as [[syntheticFloorplansDataset.FloorplanRecordBatch.permuted]] lays a target out for it.
  *
  * A prediction embedding may answer with any node the slots before it have not taken — of its
  * own class, where the nodes are written class by class — so its cost is the smallest
  * dissimilarity to any of them rather than the dissimilarity to one. The slot just past the last
  * node is charged for saying so, which is how transcription knows where to stop.
  */
class RemainingNodeLoss[V: IsFloating](vtype: VType[V], canvas: Int, written: NodeOrder)
    extends ((D2S.NodeQueryLogits[V], FloorplanNodes[Node]) => Tensor0[V]):

  /** Axis of the record's nodes seen as candidates to answer with rather than as positions. */
  private type Candidate = Prime[Node]

  override def apply(answered: D2S.NodeQueryLogits[V], target: FloorplanNodes[Node]): Tensor0[V] =
    require(answered.nodeClass.shape(Axis[PoolQuery]) == 2, "the loss pairs two answers")
    val (a, b) = (answered.at(0), answered.at(1))
    val nodes = target.nodeClass.shape.extent(Axis[Node])
    val pairs = Shape2(nodes, Axis[Candidate] -> nodes.size)
    val holdsNode = FloorplanClass.indicator(vtype)(_.isNode).slice(Axis[FloorplanClasses].at(target.nodeClass))
    val taken = holdsNode.sum
    val remaining = triu(Tensor(pairs, vtype).fill(1f)) *! holdsNode.relabelTo(Axis[Candidate])
    val candidates = written match
      case NodeOrder.Free       => remaining
      case NodeOrder.ByClass(_) =>
        val sameClass = target.nodeClass.vmap(Axis[Node])(own => target.nodeClass.relabelTo(Axis[Candidate]) elementEquals_! own)
        remaining * sameClass.asFloat(vtype)

    val asked = candidates.max(Axis[Candidate])
    val guessed = distinctly(dissimilarity(a, target), dissimilarity(b, target), candidates)
    val ended = isAt(taken, nodes, vtype)
    val stops = costOfClass(a.nodeClass, FloorplanClass.NoNode.id) + costOfClass(b.nodeClass, FloorplanClass.NoNode.id)
    // Both queries are charged, so the total is halved to stay on the scale of one answer.
    ((guessed * asked).sum + (stops * ended).sum) / ((taken + 1f) * 2f)

  /** What every position's scores would cost against every node of the record: its class and the
    * two ends of the *target* node — so that nothing depends on what the model predicts.
    */
  private def dissimilarity(logits: NodeLogits[V], target: FloorplanNodes[Node]): Tensor2[Node, Candidate, V] =
    def placed(scores: Tensor2[Node, Pixel, V], coordinate: Tensor1[Node, Float32]) =
      costOfValue(scores, Pixels.of(coordinate, canvas).relabelTo(Axis[Candidate]))
    costOfValue(logits.nodeClass, target.nodeClass.relabelTo(Axis[Candidate])) +
      placed(logits.startX, target.startX) +
      placed(logits.startY, target.startY) +
      placed(logits.endX, target.endX) +
      placed(logits.endY, target.endY)
