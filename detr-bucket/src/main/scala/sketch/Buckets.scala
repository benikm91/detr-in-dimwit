package detrbucket.sketch

import detr.*
import dataset.NodeClass
import dimwit.*

/** How the detector's queries are split by class: the first queries answer only with a node of the
  * first class or with none, the next ones only with a node of the second class, and so on.
  */
case class Buckets(sizes: Seq[(NodeClass, Int)]):

  val queries: Int = sizes.map(_._2).sum

  /** The class every query answers with, where it answers. */
  def classes: Tensor1[Query, Int32] =
    Tensor1(Axis[Query], VType[Int32]).fromArray(sizes.flatMap((nodeClass, size) => Seq.fill(size)(nodeClass.id)).toArray)

  /** Every class, with where its queries lie along [[Query]]. */
  def ranges: Seq[(NodeClass, Range)] =
    sizes.map(_._2).scanLeft(0)(_ + _).zip(sizes).map((from, bucket) => (bucket._1, from until from + bucket._2))

  override def toString: String = sizes.map((nodeClass, size) => s"$size ${nodeClass.toString.toLowerCase}").mkString(", ")
