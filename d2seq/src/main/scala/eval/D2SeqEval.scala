package d2seq.eval

import d2seq.*
import d2seq.model.*
import d2seq.train.*
import d2seq.config.*
import dataset.DrawingDataset
import dataset.DrawingDataset.Split
import dataset.Metrics
import dataset.NodeClass
import dataset.RecordBatch
import dataset.RecordGraph
import dataset.RecordNodes
import dataset.RecordScoring
import dataset.Runs
import dataset.Tolerances
import dataset.Transcripts
import deepwit.checkpointing.TensorTreeCheckpointer
import dimwit.*
import dimwit.tensor.Tensor4

import scala.language.implicitConversions

/** Scores the newest run of a setup on a whole split, and writes the metrics as one CSV, and what
  * the last checkpoint wrote down for every drawing as [[Transcripts]]. On the validation split
  * every checkpoint is scored, the last first; the test split is looked at once, by the last alone.
  */
def scoreSequenceTranscriber(setup: D2SeqSetup, size: String, split: Split): Unit =
  dimwit.initialize()

  val checkpoints = TensorTreeCheckpointer.latestIn(setup.checkpointRoot).getOrElse(sys.error(s"no training run in ${setup.checkpointRoot}"))
  println(s"reading ${checkpoints.rootPath}")
  val data = DrawingDataset.open(setup.corpus)(Axis[Width], Axis[Height], Axis[Channel], Axis[Node], Axis[Relationship])(split)
  println(s"${data.numSamples} drawings\n")

  val transcriber = Transcriber(Axis[Node] -> setup.nodeSlots, TranscribedTogether)

  /** Every drawing of the split transcribed by a model, beside the nodes of the record it was
    * rendered from — a sequence holds no relationships, so the record it is held against holds
    * none either.
    */
  def transcribed(params: D2Seq.Params[Float32]): Seq[(RecordGraph, RecordGraph)] =
    data.samples
      .grouped(TranscribedTogether)
      .flatMap(batch => batch.map(sample => RecordGraph.of(sample.target).copy(edges = Seq.empty)).zip(transcriber(params, batch.map(_.image))))
      .toSeq

  val csv = Metrics.Csv("d2seq", setup.corpus, size, split, Runs.parameters(checkpoints.loadLatest[D2SeqTrainState].get.params), Runs.trainingSeconds(checkpoints.rootPath))
  println(s"writing to ${csv.path}")

  val scored = split match
    case Split.Test => Seq(checkpoints.iterations.last)
    case _          => checkpoints.iterations.reverse
  scored.foreach: step =>
    println(s"scoring checkpoint $step")
    val drawings = transcribed(checkpoints.load[D2SeqTrainState](step).getOrElse(sys.error(s"no checkpoint $step")).params)
    val measured = Tolerances.map(tolerance => Metrics.Row(step, None, tolerance, drawings.map((target, written) => RecordScoring.score(target, written, tolerance / setup.corpus.canvas))))
    csv.append(measured)
    if step == checkpoints.iterations.last then
      measured.foreach(row => RecordScoring.reportAt(row.tolerance, row.scored))
      println(s"transcripts written to ${Transcripts.write("d2seq", setup.corpus, size, split, step, drawings)}")

/** How many drawings are transcribed together. A batch is one traced computation, so this is what
  * the split costs in calls rather than in work.
  */
private val TranscribedTogether = 32

/** Axis of the drawings transcribed together. */
private trait Drawing derives Label

/** Transcribes documents into the sequence of nodes they hold, one node at a time.
  *
  * Nothing but the document goes in, and every step reads back only what the model itself has
  * written, so no target record can reach what is written down. At every slot the node the model
  * scores highest is written down and never looked back on.
  *
  * A batch of drawings is transcribed in lockstep and as a single traced computation, so a batch
  * costs one compiled call. `drawings` is how wide that batch is; a call handing over fewer is
  * filled up with a drawing it already holds and read back short again, so that every batch is the
  * same shape. The parameters are an argument, so the one compiled computation serves every batch
  * of every checkpoint.
  *
  * A drawing writes its nodes one slot after the other, in one `scan` over the slots, and a step
  * reads what the slots before it have written from the decoder's caches rather than reading them
  * again. A drawing that has answered [[NodeClass.NoNode]] writes nothing more, and the slots it
  * would have filled hold nothing — which is what a position a record does not reach holds anyway.
  */
class Transcriber(nodes: AxisExtent[Node], drawings: Int = 1):

  private val jitTranscribe = jit(transcribe)

  private def transcribe(params: D2Seq.Params[Float32], documents: Tensor4[Drawing, Width, Height, Channel, Float32]): RecordBatch[Drawing, Node, Relationship] =
    val model = D2Seq(params)
    val slots = Tensor1(nodes.axis).fromRange(0 until nodes.size)
    val (nodeClass, construction, startX, startY, endX, endY, midX, midY) = documents.vmap(Axis[Drawing]): document =>
      val documentCache = model.read(model.encodeDocument(document))
      val (_, written) = scan(Axis[Node])((model.nothingWritten(nodes), model.start, Tensor0(true)), slots):
        case ((written, previous, writing), slot) =>
          val (logits, nowWritten) = model.answerAt(documentCache, written, slot, previous)
          val said = model.nodeHead.decide(logits)
          def only[W](values: Tensor1[Node, W]) = values.squeeze(Axis[Node])
          val stillWriting = writing and !only(said.nodeClass).elementEquals(NodeClass.NoNode.id)
          val node = (
            where(stillWriting, only(said.nodeClass), NodeClass.NoNode.id),
            where(stillWriting, only(said.construction), 0),
            where(stillWriting, only(said.startX), 0),
            where(stillWriting, only(said.startY), 0),
            where(stillWriting, only(said.endX), 0),
            where(stillWriting, only(said.endY), 0),
            where(stillWriting, only(said.midX), 0),
            where(stillWriting, only(said.midY), 0)
          )
          def one[W](value: Tensor0[W]) = value.prependAxis(Axis[Node])
          val (cls, isConstruction, sx, sy, ex, ey, mx, my) = node
          val next = model.embed(RecordNodes(one(cls), one(isConstruction), one(sx), one(sy), one(ex), one(ey), one(mx), one(my)))
          ((nowWritten, next, stillWriting), node)
      written
    val noRelationships = Tensor(Shape2(documents.shape.extent(Axis[Drawing]), Axis[Relationship] -> 0), VType[Int32]).fill(0)
    RecordBatch(nodeClass, construction, startX, startY, endX, endY, midX, midY, noRelationships, noRelationships, noRelationships)

  def apply(params: D2Seq.Params[Float32], documents: Seq[Tensor3[Width, Height, Channel, Float32]]): Seq[RecordGraph] =
    require(documents.nonEmpty, "there is nothing to transcribe")
    require(documents.size <= drawings, s"${documents.size} drawings do not fit in a batch of $drawings")
    val filled = documents.padTo(drawings, documents.last)
    RecordGraph.of(jitTranscribe(params, stack(filled, Axis[Drawing]))).take(documents.size)
