package d2gtok

import d2gtok.config.D2GTokSetup
import d2tok.floorplan.Words
import dimwit.*
import syntheticFloorplansDataset.FloorplanEdges
import syntheticFloorplansDataset.FloorplanNodes
import syntheticFloorplansDataset.FloorplanRecordBatch
import syntheticFloorplansDataset.FloorplanRelationship
import syntheticFloorplansDataset.FloorplanRelationships

import scala.language.implicitConversions

/** `records` as they are written down: the nodes in the order the d2tok run writes them, then the
  * relationships in the setup's order, each kind in a fresh random order, followed by the positions
  * they do not reach and one more, where the end is written. The nodes keep d2tok's axis.
  */
def laidOut[S: Label](records: FloorplanRecordBatch[S, d2tok.Node, d2tok.Relationship], setup: D2GTokSetup, key: Key): FloorplanRecordBatch[S, d2tok.Node, Edge] =
  val (forNodes, forEdges) = key.split2()
  val written = setup.nodeWriter.order.laidOut(records, forNodes, setup.nodeWriter.canvas)
  val kindRank = setup.relationshipOrder.kindRank
  val slots = Axis[Edge] -> setup.edgeSlots
  val (relationship, subject, obj) =
    zipvmap(Axis[S])(written.relationship, written.subject, written.obj, written.nodeClass, forEdges.splitToTensor(written.nodeClass.shape.extent(Axis[S]))):
      case (relationship, subject, obj, nodeClass, key) =>
        def endedOnEdges(named: Tensor1[d2tok.Relationship, Int32], nothing: Int) =
          val held = named.relabel(Axis[d2tok.Relationship] -> Axis[Edge])
          concatenate(held, Tensor1(Axis[Edge], VType[Int32]).fromArray(Array.fill(slots.size - held.shape(Axis[Edge]))(nothing)), Axis[Edge])
        val edges = FloorplanEdges(endedOnEdges(relationship, FloorplanRelationship.NoRelationship.id), endedOnEdges(subject, 0), endedOnEdges(obj, 0))
        val notThere = FloorplanRelationship.indicator(VType[Float32])(!_.isRelationship).slice(Axis[FloorplanRelationships].at(edges.relationship))
        val rank = edges.ranked(kindRank, nodeClass) + notThere *! (kindRank.max + 1f)
        val shuffle = Random.permutation(slots)(key.item).asFloat(VType[Float32])
        val order = (rank *! slots.size.toFloat + shuffle).argsort(Axis[Edge])
        def moved(named: Tensor1[Edge, Int32]) = named.slice(Axis[Edge].at(order))
        (moved(edges.relationship), moved(edges.subject), moved(edges.obj))
  FloorplanRecordBatch(written.nodeClass, written.startX, written.startY, written.endX, written.endY, relationship, subject, obj)

/** The nodes of a record on the axis the relationships name them along. */
def namedByEdges(nodes: FloorplanNodes[d2tok.Node]): FloorplanNodes[Node] =
  def moved[T](values: Tensor1[d2tok.Node, T]) = values.relabel(Axis[d2tok.Node] -> Axis[Node])
  FloorplanNodes(moved(nodes.nodeClass), moved(nodes.startX), moved(nodes.startY), moved(nodes.endX), moved(nodes.endY))

/** What d2tok reads at every position while it writes the `written` words: the start word, then
  * every word but the last.
  */
def readWhileWriting(written: Tensor1[Token, Int32], words: Words): Tensor1[Token, Int32] =
  concatenate(Tensor0(words.start).prependAxis(Axis[Token]), written.slice(Axis[Token].at(0 until written.shape(Axis[Token]) - 1)), Axis[Token])
