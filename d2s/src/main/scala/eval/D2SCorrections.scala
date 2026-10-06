package d2s.eval

import d2s.*
import d2s.model.*
import d2s.train.*
import d2s.config.*
import NodeHead.NodeLogits
import dataset.Canvas
import dataset.DrawingDataset
import dataset.DrawingDataset.Split
import dataset.NodeClass
import dataset.Point
import dataset.RecordGraph
import dataset.RecordNode
import dataset.RecordNodes
import dataset.Runs
import deepwit.checkpointing.TensorTreeCheckpointer
import dimwit.*

import java.nio.file.Files
import java.nio.file.Path
import scala.language.implicitConversions

/** How much help the last checkpoint of a run needs to write every validation drawing exactly,
  * when it is corrected the moment it goes wrong.
  *
  * The transcriber writes a drawing one node at a time, as [[Transcriber]] does, and every node it
  * writes is checked against the nodes of the record it has not written yet, at `tolerance` pixels:
  *
  *   - a node that matches one of them is kept, as written;
  *   - a node that matches none is replaced by one of the missing nodes, drawn at random;
  *   - stopping while nodes are missing is replaced the same way;
  *   - going on once every node is written is stopped.
  *
  * Each replacement is one help, and the transcriber carries on from the corrected record — what
  * writing the remaining nodes, given the ones already taken, allows. A drawing that needs no help
  * is one the transcriber gets exactly right on its own; one that needs exactly one is one it
  * recovers on from its first mistake.
  *
  * One line per drawing goes to `d2s-<corpus>-<size>-<step>-corrections.csv` in
  * [[Runs.outputDir]], and a summary is printed.
  */
def correctSetTranscriber(setup: D2SSetup, size: String, tolerance: Float = 4f): Unit =
  dimwit.initialize()

  val checkpoints = TensorTreeCheckpointer.latestIn(setup.checkpointRoot).getOrElse(sys.error(s"no training run in ${setup.checkpointRoot}"))
  val step = checkpoints.iterations.last
  println(s"reading checkpoint $step of ${checkpoints.rootPath}")
  val params = checkpoints.load[D2STrainState](step).getOrElse(sys.error(s"no checkpoint $step")).params
  val nodes = Axis[Node] -> setup.nodeSlots
  val data = DrawingDataset.open(setup.corpus)(Axis[Width], Axis[Height], Axis[Channel], Axis[Node], Axis[Relationship])(Split.Validation)
  println(s"correcting ${data.numSamples} drawings at $tolerance px\n")

  val encode = jit: (params: D2S.Params[Float32], documents: Tensor4[Sketch, Width, Height, Channel, Float32]) =>
    documents.vmap(Axis[Sketch])(D2S(params).encodeDocument)

  /** What every drawing writes at `slot`, given the nodes it has taken: the likeliest answer of the
    * pool, as [[Transcriber]] chooses it.
    */
  val next = jit: (params: D2S.Params[Float32], encoded: Tensor3[Sketch, Patch, Embedding, Float32], taken: Taken, slot: Tensor0[Int32]) =>
    val model = D2S(params)
    zipvmap(Axis[Sketch])(encoded, taken.nodeClass, taken.startX, taken.startY, taken.endX, taken.endY, taken.midX, taken.midY):
      case (document, nodeClass, startX, startY, endX, endY, midX, midY) =>
        val scored = model.logitsPerQuery(document, RecordNodes(nodeClass, startX, startY, endX, endY, midX, midY))
        val (saidClass, saidStartX, saidStartY, saidEndX, saidEndY, saidMidX, saidMidY, score) =
          zipvmap(Axis[PoolQuery])(scored.nodeClass, scored.startX, scored.startY, scored.endX, scored.endY, scored.midX, scored.midY):
            case (nodeClass, startX, startY, endX, endY, midX, midY) =>
              answered(model.nodeHead, NodeLogits(nodeClass, startX, startY, endX, endY, midX, midY))
        val here = Axis[Node].at(slot)
        val likeliest = Axis[PoolQuery].at(score.slice(here).argmax(Axis[PoolQuery]))
        def said[W](answers: Tensor2[PoolQuery, Node, W]) = answers.slice(here).slice(likeliest)
        (said(saidClass), said(saidStartX), said(saidStartY), said(saidEndX), said(saidEndY), said(saidMidX), said(saidMidY))

  val within = tolerance / Canvas
  def matches(target: RecordNode, written: RecordNode) =
    target.nodeClass == written.nodeClass &&
      target.points.zip(written.points).forall((wanted, at) => (wanted.x - at.x).abs <= within && (wanted.y - at.y).abs <= within)

  /** One more node of a drawing: what the transcriber wrote, kept or corrected. */
  def advance(drawing: Correcting, written: RecordNode, slot: Int): Correcting =
    // The node a correction hands over, drawn at random from the missing ones.
    def handedOver = drawing.missing(scala.util.Random(drawing.index * nodes.size + slot).nextInt(drawing.missing.size))
    def corrected(help: Help) =
      val node = handedOver
      drawing.copy(taken = drawing.taken :+ node, missing = drawing.missing.diff(Seq(node)), helps = drawing.helps :+ help)
    if drawing.isDone then drawing
    else if drawing.missing.isEmpty then
      drawing.copy(isDone = true, helps = if written.nodeClass.isNode then drawing.helps :+ Help.KeptGoing else drawing.helps)
    else if !written.nodeClass.isNode then corrected(Help.StoppedEarly(drawing.missing.size))
    else
      drawing.missing.find(matches(_, written)) match
        case Some(found) => drawing.copy(taken = drawing.taken :+ written, missing = drawing.missing.diff(Seq(found)))
        case None        => corrected(Help.WrongNode(drawing.missing.size))

  /** A batch of drawings, corrected to the end in lockstep: every drawing takes one node per slot. */
  def correctedTogether(batch: Seq[(Int, dataset.Sample[Width, Height, Channel, dataset.Record[Node, Relationship]])]): Seq[Correcting] =
    val documents = batch.map(_._2.image).padTo(CorrectedTogether, batch.last._2.image)
    val encoded = encode(params, stack(documents, Axis[Sketch]))
    val start = batch.map((index, sample) => Correcting(index, RecordGraph.of(sample.target).nodes, Vector.empty, Vector.empty, isDone = false))
    (0 until nodes.size).foldLeft(start): (drawings, slot) =>
      if drawings.forall(_.isDone) then drawings
      else
        val written = writtenAt(next(params, encoded, Taken.of(drawings.map(_.taken), nodes, CorrectedTogether), Tensor0(slot)))
        drawings.lazyZip(written).map(advance(_, _, slot))

  val corrections = data.samples.zipWithIndex.map(_.swap).grouped(CorrectedTogether).flatMap(correctedTogether).toSeq

  val path = Path.of(Runs.outputDir, s"d2s-${setup.corpus.name}-$size-$step-corrections.csv")
  Files.createDirectories(path.getParent)
  Files.writeString(
    path,
    (s"drawing,nodes,helps,wrong_nodes,stopped_early,kept_going,missing_at_first_help" +:
      corrections.map: drawing =>
        val firstHelp = drawing.helps.headOption.fold("")(_.missing.toString)
        s"${drawing.index},${drawing.size},${drawing.helps.size},${drawing.count[Help.WrongNode]},${drawing.count[Help.StoppedEarly]},${drawing.helps.count(_ == Help.KeptGoing)},$firstHelp"
    ).mkString("", "\n", "\n")
  )

  val helped = corrections.filter(_.helps.nonEmpty)
  val handedOver = corrections.map(_.helps.count(_ != Help.KeptGoing)).sum
  def percent(count: Int, of: Int) = f"${100f * count / of}%5.1f%%"
  println(f"at $tolerance px, checkpoint $step, ${corrections.size} drawings")
  println(s"  exactly right with no help        ${percent(corrections.count(_.helps.isEmpty), corrections.size)}")
  (1 to 4).foreach: helps =>
    val held = if helps < 4 then corrections.count(_.helps.size == helps) else corrections.count(_.helps.size >= helps)
    println(s"  exactly right with ${if helps < 4 then s"$helps help${if helps == 1 then " " else "s"}  " else "4+ helps"}        ${percent(held, corrections.size)}")
  println(s"  of those that need help, recover after one          ${percent(helped.count(_.helps.size == 1), helped.size)}")
  println(s"  nodes the transcriber writes itself                 ${percent(corrections.map(_.size).sum - handedOver, corrections.map(_.size).sum)}")
  // Of the nodes still missing when a drawing first needs help, the share it writes itself once
  // helped: the work a correction saves over writing the rest by hand.
  val afterFirstHelp = helped.filter(_.helps.head.missing > 0)
  val writtenAfter = afterFirstHelp.map(drawing => drawing.helps.head.missing - drawing.helps.count(_ != Help.KeptGoing)).sum
  println(s"  of the nodes missing at the first help, written by the transcriber after it ${percent(writtenAfter, afterFirstHelp.map(_.helps.head.missing).sum)}")
  println(f"  helps per helped drawing ${helped.map(_.helps.size).sum.toFloat / helped.size}%.2f, nodes missing at the first help ${afterFirstHelp.map(_.helps.head.missing).sum.toFloat / afterFirstHelp.size}%.2f")
  println(s"  helps: ${corrections.map(_.count[Help.WrongNode]).sum} wrong nodes, ${corrections.map(_.count[Help.StoppedEarly]).sum} stopped early, ${corrections.map(_.helps.count(_ == Help.KeptGoing)).sum} kept going")
  println(s"written to $path")

/** How many drawings are corrected together. */
private val CorrectedTogether = 32

/** Axis of the drawings corrected together. */
private trait Sketch derives Label

/** What a correction steps in for. `missing` is how many nodes of the record were still to be
  * written when it did.
  */
private enum Help(val missing: Int):
  case WrongNode(remaining: Int) extends Help(remaining)
  case StoppedEarly(remaining: Int) extends Help(remaining)
  case KeptGoing extends Help(0)

/** A drawing on its way: the nodes it has taken so far, the nodes of its record still missing, and
  * the help it has needed.
  */
private case class Correcting(index: Int, missing: Seq[RecordNode], taken: Vector[RecordNode], helps: Vector[Help], isDone: Boolean):

  def size: Int = missing.size + taken.size

  def count[H <: Help](using tag: scala.reflect.ClassTag[H]): Int = helps.count(tag.runtimeClass.isInstance)

/** The nodes every drawing of a batch has taken, laid out along the node slots. */
private case class Taken(
    nodeClass: Tensor2[Sketch, Node, Int32],
    startX: Tensor2[Sketch, Node, Float32],
    startY: Tensor2[Sketch, Node, Float32],
    endX: Tensor2[Sketch, Node, Float32],
    endY: Tensor2[Sketch, Node, Float32],
    midX: Tensor2[Sketch, Node, Float32],
    midY: Tensor2[Sketch, Node, Float32]
)

private object Taken:

  def of(taken: Seq[Vector[RecordNode]], nodes: AxisExtent[Node], drawings: Int): Taken =
    val filled = taken.padTo(drawings, taken.last)
    def placed(of: Point => Float, at: Int) = Tensor2(Axis[Sketch], nodes.axis, VType[Float32]).fromArray(
      filled.map(held => Array.tabulate(nodes.size)(slot => held.lift(slot).flatMap(_.points.lift(at)).fold(0f)(of))).toArray
    )
    Taken(
      nodeClass = Tensor2(Axis[Sketch], nodes.axis, VType[Int32]).fromArray(
        filled.map(held => Array.tabulate(nodes.size)(slot => held.lift(slot).fold(NodeClass.NoNode)(_.nodeClass).id)).toArray
      ),
      startX = placed(_.x, 0),
      startY = placed(_.y, 0),
      endX = placed(_.x, 1),
      endY = placed(_.y, 1),
      midX = placed(_.x, 2),
      midY = placed(_.y, 2)
    )

/** What every drawing of a batch wrote at one slot, read back to the host. */
private def writtenAt(
    said: (Tensor1[Sketch, Int32], Tensor1[Sketch, Float32], Tensor1[Sketch, Float32], Tensor1[Sketch, Float32], Tensor1[Sketch, Float32], Tensor1[Sketch, Float32], Tensor1[Sketch, Float32])
): Seq[RecordNode] =
  val (nodeClass, startX, startY, endX, endY, midX, midY) = said
  val (classes, sx, sy, ex, ey, mx, my) = (nodeClass.toArray, startX.toArray, startY.toArray, endX.toArray, endY.toArray, midX.toArray, midY.toArray)
  classes.indices.map: at =>
    val held = NodeClass.fromId(classes(at))
    RecordNode(held, Seq(Point(sx(at), sy(at)), Point(ex(at), ey(at)), Point(mx(at), my(at))).take(held.numPoints))
