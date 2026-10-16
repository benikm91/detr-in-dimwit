package syntheticFloorplansDataset

import dataset.Batch
import dataset.DrawingDataset.Split
import dataset.PythonModules
import dataset.Sample
import dataset.drawingsOf
import dimwit.*
import dimwit.jax.Jax
import dimwit.python.PyBridge.liftPyTensor
import dimwit.tensor.Tensor4
import me.shadaj.scalapy.py
import me.shadaj.scalapy.py.SeqConverters

import scala.language.implicitConversions

/** Axis over the [[FloorplanClass]] values a node of a floor plan is classified into. */
trait FloorplanClasses derives Label

/** Axis over the [[FloorplanRelationship]] values a relationship of a floor plan is classified into. */
trait FloorplanRelationships derives Label

/** What a node of a floor plan is: a wall, a door or a window, each a horizontal or vertical segment
  * placed by its two ends, read rightwards or downwards.
  *
  * [[NoNode]] marks a position the record does not reach, which is also where a transcription of its
  * nodes stops.
  */
enum FloorplanClass(val id: Int, val recordName: String):

  case NoNode extends FloorplanClass(0, "")

  /** A wall with the outside on one side, from a corner or junction to the next. */
  case OutsideWall extends FloorplanClass(1, "outside wall")

  /** A wall with rooms on both sides, from a corner or junction to the next. */
  case InsideWall extends FloorplanClass(2, "inside wall")

  /** Placed by its jambs, in an inside wall. */
  case Door extends FloorplanClass(3, "door")

  /** Placed by its jambs, in an outside wall. */
  case OutsideDoor extends FloorplanClass(4, "outside door")

  /** Placed by its jambs, in an outside wall. */
  case Window extends FloorplanClass(5, "window")

  def isNode: Boolean = this != FloorplanClass.NoNode

object FloorplanClass:

  def fromId(id: Int): FloorplanClass =
    values.find(_.id == id).getOrElse(throw IllegalArgumentException(s"unknown floor plan class id: $id"))

  def indicator[V: IsFloating](vtype: VType[V])(holds: FloorplanClass => Boolean): Tensor1[FloorplanClasses, V] =
    Tensor1(Axis[FloorplanClasses], vtype).fromArray(values.map(nodeClass => if holds(nodeClass) then 1f else 0f))

/** What a relationship of a floor plan is.
  *
  * [[NoRelationship]] marks a position the record does not reach.
  */
enum FloorplanRelationship(val id: Int, val recordName: String, val isSymmetric: Boolean):

  case NoRelationship extends FloorplanRelationship(0, "", false)

  /** Two walls sharing an end, held once with the two it links in ascending order. */
  case Joined extends FloorplanRelationship(1, "joined", true)

  /** A door, outside door or window, and the wall it stands in. */
  case In extends FloorplanRelationship(2, "in", false)

  def isRelationship: Boolean = this != FloorplanRelationship.NoRelationship

object FloorplanRelationship:

  def fromId(id: Int): FloorplanRelationship =
    values.find(_.id == id).getOrElse(throw IllegalArgumentException(s"unknown floor plan relationship id: $id"))

  def indicator[V: IsFloating](vtype: VType[V])(holds: FloorplanRelationship => Boolean): Tensor1[FloorplanRelationships, V] =
    Tensor1(Axis[FloorplanRelationships], vtype).fromArray(values.map(relationship => if holds(relationship) then 1f else 0f))

/** What a floor plan's drawing encodes: its nodes along `Node`, each a class and the two ends of a
  * segment as a fraction of the drawing's side, and its relationships along `Edge`, naming the nodes
  * they relate by their position. The positions a record does not reach hold [[FloorplanClass.NoNode]]
  * and [[FloorplanRelationship.NoRelationship]].
  */
case class FloorplanRecord[Node, Edge](
    nodeClass: Tensor1[Node, Int32],
    startX: Tensor1[Node, Float32],
    startY: Tensor1[Node, Float32],
    endX: Tensor1[Node, Float32],
    endY: Tensor1[Node, Float32],
    relationship: Tensor1[Edge, Int32],
    subject: Tensor1[Edge, Int32],
    obj: Tensor1[Edge, Int32]
)

/** [[FloorplanRecord]] for a batch of floor plans along the axis `S`. A model that predicts no
  * relationships reads the nodes alone.
  */
case class FloorplanRecordBatch[S, Node, Edge](
    nodeClass: Tensor2[S, Node, Int32],
    startX: Tensor2[S, Node, Float32],
    startY: Tensor2[S, Node, Float32],
    endX: Tensor2[S, Node, Float32],
    endY: Tensor2[S, Node, Float32],
    relationship: Tensor2[S, Edge, Int32],
    subject: Tensor2[S, Edge, Int32],
    obj: Tensor2[S, Edge, Int32]
)

/** Axis of the floor plans of a part. */
private trait Plans derives Label

/** DimWit wrapper around the synthetic floor plans
  * ([[https://huggingface.co/datasets/benikm91/synthetic-floorplans-for-transcription]]), backed by
  * ScalaPy. Its files are downloaded on first use and read from the Hugging Face cache after.
  *
  * Call `dimwit.initialize()` once before using this loader.
  *
  * {{{
  * dimwit.initialize()
  *
  * val data = SyntheticFloorplans.open(Axis[Width], Axis[Height], Axis[Channel], Axis[Node], Axis[Edge])(Split.Train)
  * val batch = data.batches(Axis[Batch] -> 64, afterSteps = 0).next()
  * }}}
  */
object SyntheticFloorplans:

  val repoId = "benikm91/synthetic-floorplans-for-transcription"

  /** The side of the drawings in pixels. */
  val canvas = 512

  /** The most nodes and relationships a record of the training split holds, which every slot is
    * sized from.
    */
  val maxNodes = 73
  val maxRelationships = 101

  /** Opens a split. A split is read in its parts — train in nine of 50,000 plans — each read when it
    * is reached, so that only one part's records are held at a time; its drawings stay memory mapped.
    */
  def open[W: Label, H: Label, C: Label, Node: Label, Edge: Label](
      width: Axis[W],
      height: Axis[H],
      channel: Axis[C],
      node: Axis[Node],
      edge: Axis[Edge]
  )(split: Split): SyntheticFloorplans[W, H, C, Node, Edge] =
    val parts = module.parts(repoId, split.fileName).as[Seq[py.Dynamic]].map: part =>
      (part.applyDynamic("__getitem__")(0).as[String], part.applyDynamic("__getitem__")(1).as[Int])
    require(parts.nonEmpty, s"no part of the split '${split.fileName}' in $repoId")
    new SyntheticFloorplans(parts)

  private[syntheticFloorplansDataset] lazy val module: py.Dynamic =
    PythonModules("synthetic_floorplans", "./syntheticFloorplansDataset/src/main/resources/python")

/** A single split of the synthetic floor plans. Use [[SyntheticFloorplans.open]] to create one. */
final class SyntheticFloorplans[W: Label, H: Label, C: Label, Node: Label, Edge: Label] private[syntheticFloorplansDataset] (
    /** Every part of the split, by name, and how many plans it holds, in seed order. */
    val parts: Seq[(String, Int)]
):
  import SyntheticFloorplans.*

  val numSamples: Int = parts.map(_._2).sum

  /** Every floor plan of the split, once, in seed order. */
  def samples: Iterator[Sample[W, H, C, FloorplanRecord[Node, Edge]]] =
    parts.indices.iterator.flatMap: part =>
      val (images, parsed) = read(part)
      val records = recordsIn(parsed, Axis[Plans] -> parts(part)._2, 0)
      (0 until parts(part)._2).iterator.map: at =>
        val plan = Axis[Plans].at(at)
        val image = drawingsOf(liftPyTensor[(Plans, H, W), UInt8](Jax.jnp.asarray(images.applyDynamic("__getitem__")(py.Dynamic.global.slice(at, at + 1)))), Axis[C])
        Sample(
          image.squeeze(Axis[Plans]),
          FloorplanRecord(
            records.nodeClass.slice(plan),
            records.startX.slice(plan),
            records.startY.slice(plan),
            records.endX.slice(plan),
            records.endY.slice(plan),
            records.relationship.slice(plan),
            records.subject.slice(plan),
            records.obj.slice(plan)
          )
        )

  /** Batches of floor plans, for as long as they are asked for, on the host: the parts in turn, each
    * cut into whole batches. The plans were generated independently of one another, so reading them
    * in order is already a shuffle.
    *
    * @param afterSteps The batches already taken, which are passed over without being read: a run
    *                   that carries on from a checkpoint reads on where it left off, and the parts
    *                   it passes are not read at all.
    */
  def batches[S: Label](batch: AxisExtent[S], afterSteps: Int): Iterator[Batch[S, W, H, FloorplanRecordBatch[S, Node, Edge]]] =
    val perPart = parts.map(_._2 / batch.size)
    require(perPart.forall(_ > 0), s"a batch of ${batch.size} exceeds a part of ${parts.map(_._2).min} plans")
    val skipped = afterSteps % perPart.sum
    val firstPart = perPart.scanLeft(0)(_ + _).init.lastIndexWhere(_ <= skipped)
    val intoFirst = skipped - perPart.take(firstPart).sum
    Iterator.from(firstPart).flatMap: at =>
      val part = at % parts.size
      val (images, parsed) = read(part)
      val starts = (0 until perPart(part)).map(_ * batch.size).drop(if at == firstPart then intoFirst else 0)
      starts.iterator.map: from =>
        val pixels = images.applyDynamic("__getitem__")(py.Dynamic.global.slice(from, from + batch.size))
        Batch(liftPyTensor[(S, H, W), UInt8](pixels), recordsIn(parsed, batch, from))

  override def toString: String = s"SyntheticFloorplans($repoId, plans=$numSamples, parts=${parts.size})"

  /** A part's drawings, memory mapped, and its records, parsed on the host. */
  private def read(part: Int): (py.Dynamic, py.Dynamic) =
    val name = parts(part)._1
    (
      module.drawings(repoId, name),
      module.records(repoId, name, canvas, maxNodes, maxRelationships,
        FloorplanClass.values.toSeq.map(_.recordName).toPythonCopy, FloorplanRelationship.values.toSeq.map(_.recordName).toPythonCopy)
    )

  /** The records of the plans from `from` on, cut from the parsed ones. */
  private def recordsIn[S: Label](parsed: py.Dynamic, rows: AxisExtent[S], from: Int): FloorplanRecordBatch[S, Node, Edge] =
    val taken = py.Dynamic.global.slice(from, from + rows.size)
    def cut(at: Int) = parsed.applyDynamic("__getitem__")(at).applyDynamic("__getitem__")(taken)
    FloorplanRecordBatch(
      nodeClass = liftPyTensor[(S, Node), Int32](cut(0)),
      startX = liftPyTensor[(S, Node), Float32](cut(1)),
      startY = liftPyTensor[(S, Node), Float32](cut(2)),
      endX = liftPyTensor[(S, Node), Float32](cut(3)),
      endY = liftPyTensor[(S, Node), Float32](cut(4)),
      relationship = liftPyTensor[(S, Edge), Int32](cut(5)),
      subject = liftPyTensor[(S, Edge), Int32](cut(6)),
      obj = liftPyTensor[(S, Edge), Int32](cut(7))
    )
