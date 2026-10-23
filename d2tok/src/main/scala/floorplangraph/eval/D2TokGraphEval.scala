package d2tok.floorplangraph.eval

import d2tok.*
import d2tok.floorplan.Words
import d2tok.floorplangraph.*
import d2tok.floorplangraph.config.*
import d2tok.floorplangraph.model.*
import d2tok.floorplangraph.train.*
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

/** What a run writes its metrics and transcripts as: d2tok, writing the relationships too. */
val Model = "d2tok-graph"

/** Scores the newest run of a setup on a whole split, and writes the metrics as one CSV, and what
  * the last checkpoint wrote down for every plan as [[FloorplanTranscripts]]. On the validation split
  * every checkpoint is scored, the last first; the test split is looked at once, by the last alone.
  */
def scoreGraphTranscriber(setup: D2TokGraphSetup, size: String, split: Split): Unit =
  dimwit.initialize()

  val checkpoints = TensorTreeCheckpointer.latestIn(setup.checkpointRoot).getOrElse(sys.error(s"no training run in ${setup.checkpointRoot}"))
  println(s"reading ${checkpoints.rootPath}")
  val data = SyntheticFloorplans.open(Axis[Width], Axis[Height], Axis[Channel], Axis[Node], Axis[Relationship])(split, setup.canvas)
  println(s"${data.numSamples} plans\n")

  val transcriber = Transcriber(setup, TranscribedTogether)

  /** Every plan of the split transcribed by a model, beside the record it was drawn from. */
  def transcribed(params: D2TokGraph.Params[Float32]): Seq[(FloorplanGraph, FloorplanGraph)] =
    data.samples
      .grouped(TranscribedTogether)
      .flatMap(batch => batch.map(sample => FloorplanGraph.of(sample.target)).zip(transcriber(params, batch.map(_.image))))
      .toSeq

  val csv = Metrics.Csv(Model, SyntheticFloorplans.name, size, split, Runs.parameters(checkpoints.loadLatest[D2TokGraphTrainState].get.params), Runs.trainingSeconds(checkpoints.rootPath))
  println(s"writing to ${csv.path}")

  val scored = split match
    case Split.Test => Seq(checkpoints.iterations.last)
    case _          => checkpoints.iterations.reverse
  scored.foreach: step =>
    println(s"scoring checkpoint $step")
    val plans = transcribed(checkpoints.load[D2TokGraphTrainState](step).getOrElse(sys.error(s"no checkpoint $step")).params)
    val measured = Tolerances.map(tolerance => Metrics.Row(step, None, tolerance, plans.map((target, written) => FloorplanScoring.score(target, written, tolerance / setup.canvas))))
    csv.append(measured)
    if step == checkpoints.iterations.last then
      measured.foreach(row => RecordScoring.reportAt(row.tolerance, row.scored))
      println(s"transcripts written to ${FloorplanTranscripts.write(Model, size, split, step, setup.canvas, plans)}")

/** How many drawings are transcribed together. A batch is one traced computation, so this is what
  * the split costs in calls rather than in work.
  */
private val TranscribedTogether = 32

/** Axis of the drawings transcribed together. */
private trait Drawing derives Label

/** Transcribes documents into the words of their records, the nodes' and then the relationships',
  * one word at a time, and reads the words back as a graph.
  *
  * Nothing but the document goes in, and every step reads back only what the model itself has
  * written. At every position the word the model scores highest is written down, whatever word it
  * is: the model is held to the language only when its words are read, up to their end or to the
  * first word that does not fit where it stands. The relationships read the nodes the model wrote:
  * those before the first end among the node words.
  *
  * A batch of drawings is transcribed in lockstep and as a single traced computation, in one `scan`
  * over every position the node words can take and one over every position the relationship words
  * can, with the words before a position read from the decoders' caches. `drawings` is how wide that
  * batch is; a call handing over fewer is filled up with a drawing it already holds and read back
  * short again, so that every batch is the same shape.
  */
class Transcriber(setup: D2TokGraphSetup, drawings: Int = 1):

  private val words: Words = setup.words
  private val relationshipWords: RelationshipWords = setup.relationshipWords
  private val tokens = Axis[Token] -> setup.tokens
  private val relationshipTokens = Axis[RelationshipToken] -> setup.relationshipTokens

  private val jitTranscribe = jit(transcribe)

  private def transcribe(params: D2TokGraph.Params[Float32], documents: Tensor4[Drawing, Width, Height, Channel, Float32]): (Tensor2[Drawing, Token, Int32], Tensor2[Drawing, RelationshipToken, Int32]) =
    val model = D2TokGraph(params)
    val positions = Tensor1(tokens.axis).fromRange(0 until tokens.size)
    val relationshipPositions = Tensor1(relationshipTokens.axis).fromRange(0 until relationshipTokens.size)
    documents.vmap(Axis[Drawing]): document =>
      val encoded = model.encodeDocument(document)
      val documentCache = model.nodes.read(encoded)
      val (_, nodeWords) = scan(Axis[Token])((model.nodes.nothingWritten(tokens), Tensor0(words.start)), positions):
        case ((written, previous), position) =>
          val (scores, nowWritten) = model.nodes.answerAt(documentCache, written, position, previous)
          val word = scores.argmax(Axis[Word])
          ((nowWritten, word), word)
      val read = concatenate(Tensor0(words.start).prependAxis(Axis[Token]), nodeWords.slice(Axis[Token].at(0 until tokens.size - 1)), Axis[Token])
      val firstEnd = where(nodeWords elementEquals_! words.end, positions, Tensor.like(positions).fill(tokens.size)).min
      val numbers = Tensor1(Axis[Node]).fromRange(0 until (tokens.size - 1) / words.perNode)
      val present = (numbers *! words.perNode +! words.perNode) <=! firstEnd
      val sources = model.read(encoded, model.numbered(model.nodes.carried(encoded, read), words.perNode, present), present)
      val (_, relationshipWordsWritten) = scan(Axis[RelationshipToken])((model.nothingWritten(relationshipTokens), Tensor0(relationshipWords.start)), relationshipPositions):
        case ((written, previous), position) =>
          val (scores, nowWritten) = model.answerAt(sources, written, position, previous)
          val word = scores.argmax(Axis[Word])
          ((nowWritten, word), word)
      (nodeWords, relationshipWordsWritten)

  /** The words the model writes for every document, the nodes' and the relationships', the
    * positions after their ends among them.
    */
  def written(params: D2TokGraph.Params[Float32], documents: Seq[Tensor3[Width, Height, Channel, Float32]]): Seq[(Seq[Int], Seq[Int])] =
    require(documents.nonEmpty, "there is nothing to transcribe")
    require(documents.size <= drawings, s"${documents.size} drawings do not fit in a batch of $drawings")
    val filled = documents.padTo(drawings, documents.last)
    val (nodeWords, relationshipWordsWritten) = jitTranscribe(params, stack(filled, Axis[Drawing]))
    nodeWords.toArray.map(_.toSeq).toSeq.zip(relationshipWordsWritten.toArray.map(_.toSeq).toSeq).take(documents.size)

  /** The records the written words read as. */
  def apply(params: D2TokGraph.Params[Float32], documents: Seq[Tensor3[Width, Height, Channel, Float32]]): Seq[FloorplanGraph] =
    written(params, documents).map((nodeWords, saidRelationships) => FloorplanGraph(words.read(nodeWords), relationshipWords.read(saidRelationships)))
