package d2gtok.eval

import d2gtok.*
import d2gtok.config.*
import d2gtok.model.*
import d2gtok.train.SavedParams
import d2g.model.EdgeSources
import d2g.model.NodeSource
import d2s.eval.chosen
import dataset.DrawingDataset.Split
import dataset.Metrics
import dataset.RecordScoring
import dataset.Runs
import dataset.Tolerances
import deepwit.attention.KVCache
import deepwit.checkpointing.TensorTreeCheckpointer
import dimwit.*
import dimwit.tensor.Tensor4
import syntheticFloorplansDataset.FloorplanEdge
import syntheticFloorplansDataset.FloorplanEdges
import syntheticFloorplansDataset.FloorplanGraph
import syntheticFloorplansDataset.FloorplanRelationship
import syntheticFloorplansDataset.FloorplanScoring
import syntheticFloorplansDataset.FloorplanTranscripts
import syntheticFloorplansDataset.SyntheticFloorplans

import scala.language.implicitConversions

/** Scores the newest run of a setup on a whole split, and writes the metrics as one CSV, and what
  * the last checkpoint wrote down for every plan as [[FloorplanTranscripts]]. On the validation split
  * every checkpoint is scored, the last first; the test split is looked at once, by the last alone.
  */
def scoreTranscriber(setup: D2GTokSetup, size: String, split: Split): Unit =
  dimwit.initialize()

  val checkpoints = TensorTreeCheckpointer.latestIn(setup.checkpointRoot).getOrElse(sys.error(s"no training run in ${setup.checkpointRoot}"))
  println(s"reading ${checkpoints.rootPath}")
  val data = SyntheticFloorplans.open(Axis[Width], Axis[Height], Axis[Channel], Axis[Node], Axis[Edge])(setup.corpus, split, setup.nodeWriter.canvas)
  println(s"pool of ${setup.queryPool} queries, ${data.numSamples} plans\n")

  val transcriber = Transcriber(setup, TranscribedTogether)

  /** Every plan of the split transcribed by a model, beside the record it was drawn from. */
  def transcribed(params: D2GTok.Params[Float32]): Seq[(FloorplanGraph, FloorplanGraph)] =
    data.samples
      .grouped(TranscribedTogether)
      .flatMap(batch => batch.map(sample => FloorplanGraph.of(sample.target)).zip(transcriber(params, batch.map(_.image))))
      .toSeq

  val csv = Metrics.Csv("d2gtok", setup.corpus.name, size, split, Runs.parameters(checkpoints.loadLatest[SavedParams].get.params), Runs.trainingSeconds(checkpoints.rootPath))
  println(s"writing to ${csv.path}")

  val scored = split match
    case Split.Test => Seq(checkpoints.iterations.last)
    case _          => checkpoints.iterations.reverse
  scored.foreach: step =>
    println(s"scoring checkpoint $step")
    val plans = transcribed(checkpoints.load[SavedParams](step).getOrElse(sys.error(s"no checkpoint $step")).params)
    val measured = Tolerances.map(tolerance => Metrics.Row(step, None, tolerance, plans.map((target, written) => FloorplanScoring.score(target, written, tolerance / setup.nodeWriter.canvas))))
    csv.append(measured)
    if step == checkpoints.iterations.last then
      measured.foreach(row => RecordScoring.reportAt(row.tolerance, row.scored))
      println(s"transcripts written to ${FloorplanTranscripts.write("d2gtok", setup.corpus, size, split, step, setup.nodeWriter.canvas, plans)}")

/** How many plans are transcribed together. A batch is one traced computation, so this is what
  * the split costs in calls rather than in work.
  */
private val TranscribedTogether = 32

/** Axis of the plans transcribed together. */
private trait Drawing derives Label

/** Transcribes floor plans: the nodes in d2tok's words, as [[d2tok.floorplan.eval.Transcriber]]
  * writes them, then the relationships between them as [[d2g.floorplan.eval.Transcriber]] writes
  * its own, reading the nodes as d2tok carried the words it wrote.
  *
  * Nothing but the document goes in, and every step reads back only what the model itself has
  * written. A node is there when all its words come before the first end among them; the nodes
  * read back stop at a word that does not fit, and a relationship naming a node past them is
  * dropped.
  *
  * `drawings` is how many plans are transcribed in one traced computation; a call handing over fewer
  * is filled up with a plan it already holds and read back short again.
  */
class Transcriber(setup: D2GTokSetup, drawings: Int = 1):

  private val words = setup.nodeWriter.words
  private val tokens = Axis[Token] -> setup.nodeWriter.tokens
  private val nodes = Axis[Node] -> setup.nodeSlots
  private val edges = Axis[Edge] -> setup.edgeSlots

  private val writeNodes = d2tok.floorplan.eval.Transcriber(words, tokens, drawings)
  private val jitRelate = jit(relate)

  /** The relationship, subject and object written at every slot of every plan. */
  private def relate(params: D2GTok.Params[Float32], documents: Tensor4[Drawing, Width, Height, Channel, Float32], written: Tensor2[Drawing, Token, Int32]): (Tensor2[Drawing, Edge, Int32], Tensor2[Drawing, Edge, Int32], Tensor2[Drawing, Edge, Int32]) =
    val model = D2GTok(params)
    val positions = Tensor1(tokens.axis).fromRange(0 until tokens.size)
    val numbers = Tensor1(nodes.axis).fromRange(0 until nodes.size)
    val edgeSlots = Tensor1(edges.axis).fromRange(0 until edges.size)
    zipvmap(Axis[Drawing])(documents, written):
      case (document, written) =>
        val encoded = model.nodes.encodeDocument(document)
        val firstEnd = where(written elementEquals_! words.end, positions, Tensor.like(positions).fill(tokens.size)).min
        val present = (numbers *! words.perNode +! words.perNode) <=! firstEnd
        val writtenNodes = model.nodesOf(model.nodes.carried(encoded, readWhileWriting(written, words)), present)
        val sources = model.readForEdges(encoded, writtenNodes)
        val (_, edge) = scan(Axis[Edge])((model.noEdgeTaken(edges), Tensor0(true)), edgeSlots):
          case ((taken, writing), slot) =>
            val (edge, stillWriting) = answeredEdgeAt(model, writtenNodes, sources, taken, slot, writing)
            def one[W](value: Tensor0[W]) = value.prependAxis(Axis[Edge])
            val (related, sub, ob) = edge
            ((model.takeEdge(writtenNodes, sources, taken, slot, FloorplanEdges(one(related), one(sub), one(ob))), stillWriting), edge)
        edge

  /** The likeliest of what the pool answers at relationship `slot` — the relationship a plan that
    * is still `writing` takes there, nothing for one that is not — and whether it is still writing
    * after it.
    */
  private def answeredEdgeAt(model: D2GTok[Float32], nodes: NodeSource[Embedding, Float32], sources: List[EdgeSources[Float32]], taken: List[KVCache[Edge, Float32]], slot: Tensor0[Int32], writing: Tensor0[Bool]) =
    val candidates = model.answerEdgeAt(nodes, sources, taken, slot)
    val said = model.edgeHead.decide(candidates)
    val score = chosen(candidates.relationship) + chosen(candidates.subject) + chosen(candidates.obj)
    val likeliest = Axis[Edge].at(score.argmax(Axis[Edge]))
    val stillWriting = writing and !said.relationship.slice(likeliest).elementEquals(FloorplanRelationship.NoRelationship.id)
    val edge = (
      where(stillWriting, said.relationship.slice(likeliest), FloorplanRelationship.NoRelationship.id),
      where(stillWriting, said.subject.slice(likeliest), 0),
      where(stillWriting, said.obj.slice(likeliest), 0)
    )
    (edge, stillWriting)

  def apply(params: D2GTok.Params[Float32], documents: Seq[Tensor3[Width, Height, Channel, Float32]]): Seq[FloorplanGraph] =
    require(documents.nonEmpty, "there is nothing to transcribe")
    require(documents.size <= drawings, s"${documents.size} plans do not fit in a batch of $drawings")
    val filled = documents.padTo(drawings, documents.last)
    val said = writeNodes.written(params.nodes, filled)
    val writtenWords = stack(said.map(plan => Tensor1(tokens.axis, VType[Int32]).fromArray(plan.toArray)), Axis[Drawing])
    val (related, subjects, objs) = jitRelate(params, stack(filled, Axis[Drawing]), writtenWords)
    val (relationship, subject, obj) = (related.toArray, subjects.toArray, objs.toArray)
    documents.indices.map: plan =>
      val nodesRead = words.read(said(plan))
      val edgesRead = relationship(plan).indices.flatMap: slot =>
        val (related, from, to) = (FloorplanRelationship.fromId(relationship(plan)(slot)), subject(plan)(slot), obj(plan)(slot))
        Option.when(related.isRelationship && from < nodesRead.size && to < nodesRead.size)(FloorplanEdge(related, from, to))
      FloorplanGraph(nodesRead, edgesRead)
