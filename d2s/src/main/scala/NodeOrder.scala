package d2s

import dataset.NodeClass
import dataset.NodeClasses
import dimwit.*

/** The order a record's nodes are written down in, one at a time. */
enum NodeOrder:

  /** Any order: a record's nodes are a set. */
  case Free

  /** Class by class, in the order `classes` lists them, and in any order within a class. */
  case ByClass(classes: Seq[NodeClass])

  /** The rank of every class, by which [[dataset.RecordBatch.permuted]] lays a target out in this
    * order: the classes not listed, [[NodeClass.NoNode]] among them, after the ones that are.
    */
  def classRank: Tensor1[NodeClasses, Float32] = this match
    case Free             => NodeClass.indicator(VType[Float32])(!_.isNode)
    case ByClass(classes) =>
      Tensor1(Axis[NodeClasses], VType[Float32]).fromArray(NodeClass.values.map(nodeClass => classes.indexOf(nodeClass) match
        case -1 => classes.size.toFloat
        case at => at.toFloat
      ))

object NodeOrder:

  /** The simplest class first: what a node needs to be placed by grows from one point to three,
    * and a node written later can be placed against the ones before it.
    */
  val SimplestFirst: NodeOrder =
    ByClass(Seq(NodeClass.Point, NodeClass.Line, NodeClass.Circle, NodeClass.Arc, NodeClass.Annotation))
