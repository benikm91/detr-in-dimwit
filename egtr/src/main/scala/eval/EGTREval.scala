package egtr.eval

import detr.*
import detr.model.*
import detr.train.*
import detr.eval.*
import detr.config.*
import egtr.*
import egtr.model.*
import egtr.train.*
import egtr.config.*
import dataset.Canvas
import dataset.Corpus
import dataset.DrawingDataset
import dataset.DrawingDataset.Split
import dataset.Objects
import dataset.RecordGraph
import dataset.Runs
import dataset.Metrics
import dataset.RecordScoring
import dataset.Tolerances
import dataset.report
import deepwit.checkpointing.TensorTreeCheckpointer
import dimwit.*

/** Scores a trained scene graph model on the whole validation split of the corpus its setup
  * names.
  *
  * What the model predicts is read back into the record it stands for and compared with the
  * record the drawing was rendered from, by [[RecordScoring]] — the same comparison
  * [[scoreTranscriber]] and [[scoreDetector]] make, through the same code. Transcribing a document into a
  * graph and detecting objects and relating them are the same task, so they are scored the same
  * way: nodes matched by what they say rather than by the slot they sit in, relationships
  * compared by the nodes they name.
  *
  * The queries a scene graph model works in do not survive that reading: a query holding no
  * object is no node, and the relations it carries relate nothing.
  */
def scoreSceneGraph(setup: EGTRSetup, size: String): Unit =
  dimwit.initialize()

  val checkpoints = TensorTreeCheckpointer.latestIn(setup.checkpointRoot).getOrElse(sys.error(s"no training run in ${setup.checkpointRoot}"))
  println(s"reading ${checkpoints.rootPath}")
  val data = DrawingDataset.open(setup.corpus)(Axis[Width], Axis[Height], Axis[Channel], Axis[BoundingBox], Axis[Relationship])(Split.Validation)

  /** Every drawing of the split beside the graph a model settles on at each threshold: the model
    * is read once per drawing, and the relationships it claims are cut from that at every one.
    */
  def predicted(params: EGTR.Params[Float32]): Seq[(RecordGraph, Map[Float, RecordGraph])] =
    val predict = jit(EGTR(params).apply)
    data.samples
      .map: sample =>
        val graph = predict(sample.image)
        def claimed(threshold: Float) =
          RecordGraph.of(Objects(graph.objects, (graph.relations > Tensor.like(graph.relations).fill(threshold)).asFloat(VType[Float32])))
        (RecordGraph.of(sample.target), Thresholds.map(threshold => threshold -> claimed(threshold)).toMap)
      .toSeq

  val rows = checkpoints.iterations.flatMap: step =>
    println(s"scoring checkpoint $step")
    val drawings = predicted(checkpoints.load[EGTRTrainState](step).getOrElse(sys.error(s"no checkpoint $step")).params)
    for
      threshold <- Thresholds
      tolerance <- Tolerances
    yield Metrics.Row(step, Some(threshold), tolerance, drawings.map((target, at) => RecordScoring.score(target, at(threshold), tolerance / Canvas)))

  rows.filter(row => row.step == checkpoints.iterations.last && row.threshold.contains(0.5f)).foreach(row => RecordScoring.reportAt(row.tolerance, row.scored))
  val params = checkpoints.loadLatest[EGTRTrainState].get.params
  println(s"written to ${Metrics.write("egtr", setup.corpus, size, Runs.parameters(params), Runs.trainingSeconds(checkpoints.rootPath), rows)}")

/** Above which score a relation counts as claimed. A record is a set of relationships, not a
  * ranking of them, so a relation has to be either in or out — and where the line goes is a
  * choice the model does not make, so it is scored at every one of these.
  */
private val Thresholds = Seq(0.9f, 0.75f, 0.5f, 0.25f, 0.1f)
