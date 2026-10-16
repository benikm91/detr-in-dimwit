package d2s.sketch.train

import d2s.*
import d2s.sketch.*
import d2s.model.*
import d2s.sketch.model.*
import d2s.train.*
import dataset.NodeClass
import dataset.NodeClasses
import dataset.RecordNodes
import NodeHead.NodeLogits
import dimwit.*

import scala.language.implicitConversions

/** What a record's nodes cost the model that writes them down, in the order they are `written` in,
  * as [[dataset.RecordBatch.permuted]] lays a target out for it.
  *
  * A prediction embedding may answer with any node the slots before it have not taken — of its
  * own class, where the nodes are written class by class — so its cost is the smallest
  * dissimilarity to any of them rather than the dissimilarity to one. The slot just past the last
  * node is charged for saying so, which is how transcription knows where to stop.
  */
class RemainingNodeLoss[V: IsFloating](vtype: VType[V], canvas: Int, written: NodeOrder)
    extends ((D2S.NodeQueryLogits[V], RecordNodes[Node]) => Tensor0[V]):

  /** Axis of the record's nodes seen as candidates to answer with rather than as positions. */
  private type Candidate = Prime[Node]

  override def apply(answered: D2S.NodeQueryLogits[V], target: RecordNodes[Node]): Tensor0[V] =
    require(answered.nodeClass.shape(Axis[PoolQuery]) == 2, "the loss pairs two answers")
    val (a, b) = (answered.at(0), answered.at(1))
    val nodes = target.nodeClass.shape.extent(Axis[Node])
    val pairs = Shape2(nodes, Axis[Candidate] -> nodes.size)
    val holdsNode = NodeClass.indicator(vtype)(_.isNode).slice(Axis[NodeClasses].at(target.nodeClass))
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
    val stops = costOfClass(a.nodeClass, NodeClass.NoNode.id) + costOfClass(b.nodeClass, NodeClass.NoNode.id)
    // Both queries are charged, so the total is halved to stay on the scale of one answer.
    ((guessed * asked).sum + (stops * ended).sum) / ((taken + 1f) * 2f)

  /** What every position's scores would cost against every node of the record: its class, whether
    * it is construction geometry, and where the *target* node is placed — so that nothing depends
    * on what the model predicts. A class that runs nowhere is not measured on where it ends, nor
    * one that does not bend on its middle.
    */
  private def dissimilarity(logits: NodeLogits[V], target: RecordNodes[Node]): Tensor2[Node, Candidate, V] =
    val candidateClass = target.nodeClass.relabelTo(Axis[Candidate])
    def placed(scores: Tensor2[Node, Pixel, V], coordinate: Tensor1[Node, Float32]) =
      costOfValue(scores, Pixels.of(coordinate, canvas).relabelTo(Axis[Candidate]))
    val runsOn = NodeClass.indicator(vtype)(_.numPoints > 1).slice(Axis[NodeClasses].at(candidateClass))
    val bends = NodeClass.indicator(vtype)(_.numPoints > 2).slice(Axis[NodeClasses].at(candidateClass))
    val ends = placed(logits.endX, target.endX) + placed(logits.endY, target.endY)
    val middles = placed(logits.midX, target.midX) + placed(logits.midY, target.midY)
    costOfValue(logits.nodeClass, candidateClass) +
      costOfValue(logits.construction, target.construction.relabelTo(Axis[Candidate])) +
      placed(logits.startX, target.startX) +
      placed(logits.startY, target.startY) +
      ends *! runsOn +
      middles *! bends
