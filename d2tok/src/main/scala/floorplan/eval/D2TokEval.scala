package d2tok.floorplan.eval

import d2tok.*
import d2tok.model.*
import d2tok.floorplan.*
import d2tok.floorplan.config.*
import d2tok.floorplan.train.*
import dataset.DrawingDataset.Split
import dataset.Metrics
import dataset.RecordScoring
import dataset.Runs
import dataset.Tolerances
import deepwit.checkpointing.TensorTreeCheckpointer
import dimwit.*
import dimwit.tensor.Tensor4

import syntheticFloorplansDataset.FloorplanGraph
import syntheticFloorplansDataset.FloorplanScoring
import syntheticFloorplansDataset.FloorplanTranscripts
import syntheticFloorplansDataset.SyntheticFloorplans

import scala.language.implicitConversions

/** Scores the newest run of a setup on a whole split, and writes the metrics as one CSV, and what
  * the last checkpoint wrote down for every plan as [[FloorplanTranscripts]]. On the validation split
  * every checkpoint is scored, the last first; the test split is looked at once, by the last alone.
  */
def scoreWordTranscriber(setup: D2TokSetup, size: String, split: Split): Unit =
  dimwit.initialize()

  val checkpoints = TensorTreeCheckpointer.latestIn(setup.checkpointRoot).getOrElse(sys.error(s"no training run in ${setup.checkpointRoot}"))
  println(s"reading ${checkpoints.rootPath}")
  val data = SyntheticFloorplans.open(Axis[Width], Axis[Height], Axis[Channel], Axis[Node], Axis[Relationship])(split, setup.canvas)
  println(s"${data.numSamples} plans\n")

  val transcriber = Transcriber(setup.words, Axis[Token] -> setup.tokens, TranscribedTogether)

  /** Every plan of the split transcribed by a model, beside the nodes of the record it was drawn
    * from — a sequence of words holds no relationships, so the record it is held against holds
    * none either.
    */
  def transcribed(params: D2Tok.Params[Float32]): Seq[(FloorplanGraph, FloorplanGraph)] =
    data.samples
      .grouped(TranscribedTogether)
      .flatMap(batch => batch.map(sample => FloorplanGraph.of(sample.target).copy(edges = Seq.empty)).zip(transcriber(params, batch.map(_.image))))
      .toSeq

  val csv = Metrics.Csv("d2tok", SyntheticFloorplans.name, size, split, Runs.parameters(checkpoints.loadLatest[D2TokTrainState].get.params), Runs.trainingSeconds(checkpoints.rootPath))
  println(s"writing to ${csv.path}")

  val scored = split match
    case Split.Test => Seq(checkpoints.iterations.last)
    case _          => checkpoints.iterations.reverse
  scored.foreach: step =>
    println(s"scoring checkpoint $step")
    val drawings = transcribed(checkpoints.load[D2TokTrainState](step).getOrElse(sys.error(s"no checkpoint $step")).params)
    val measured = Tolerances.map(tolerance => Metrics.Row(step, None, tolerance, drawings.map((target, written) => FloorplanScoring.score(target, written, tolerance / setup.canvas))))
    csv.append(measured)
    if step == checkpoints.iterations.last then
      measured.foreach(row => RecordScoring.reportAt(row.tolerance, row.scored))
      println(s"transcripts written to ${FloorplanTranscripts.write("d2tok", size, split, step, setup.canvas, drawings)}")

/** How many drawings are transcribed together. A batch is one traced computation, so this is what
  * the split costs in calls rather than in work.
  */
private val TranscribedTogether = 32

/** Axis of the drawings transcribed together. */
private trait Drawing derives Label

/** Transcribes documents into the words of their records, one word at a time, and reads the words
  * back as nodes.
  *
  * Nothing but the document goes in, and every step reads back only what the model itself has
  * written. At every position the word the model scores highest is written down, whatever word it
  * is: the model is held to the language only when its words are read, up to their end or to the
  * first word that does not fit where it stands.
  *
  * A batch of drawings is transcribed in lockstep and as a single traced computation, in one `scan`
  * over every position a record's words can take, with the words before a position read from the
  * decoder's caches. `drawings` is how wide that batch is; a call handing over fewer is filled up
  * with a drawing it already holds and read back short again, so that every batch is the same shape.
  */
class Transcriber(words: Words, tokens: AxisExtent[Token], drawings: Int = 1):

  private val jitTranscribe = jit(transcribe)

  private def transcribe(params: D2Tok.Params[Float32], documents: Tensor4[Drawing, Width, Height, Channel, Float32]): Tensor2[Drawing, Token, Int32] =
    val model = D2Tok(params)
    val positions = Tensor1(tokens.axis).fromRange(0 until tokens.size)
    documents.vmap(Axis[Drawing]): document =>
      val documentCache = model.read(model.encodeDocument(document))
      val (_, written) = scan(Axis[Token])((model.nothingWritten(tokens), Tensor0(words.start)), positions):
        case ((written, previous), position) =>
          val (scores, nowWritten) = model.answerAt(documentCache, written, position, previous)
          val word = scores.argmax(Axis[Word])
          ((nowWritten, word), word)
      written

  /** The words the model writes for every document, the positions after its end among them. */
  def written(params: D2Tok.Params[Float32], documents: Seq[Tensor3[Width, Height, Channel, Float32]]): Seq[Seq[Int]] =
    require(documents.nonEmpty, "there is nothing to transcribe")
    require(documents.size <= drawings, s"${documents.size} drawings do not fit in a batch of $drawings")
    val filled = documents.padTo(drawings, documents.last)
    jitTranscribe(params, stack(filled, Axis[Drawing])).toArray.take(documents.size).map(_.toSeq).toSeq

  /** The records the written words read as. */
  def apply(params: D2Tok.Params[Float32], documents: Seq[Tensor3[Width, Height, Channel, Float32]]): Seq[FloorplanGraph] =
    written(params, documents).map(said => FloorplanGraph(words.read(said), Seq.empty))
