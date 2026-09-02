package egtr

import dataset.Canvas
import dataset.Corpus
import dataset.DrawingDataset
import dataset.DrawingDataset.Split
import dataset.Objects
import dataset.RecordGraph
import dataset.RecordScoring
import dataset.Tolerances
import dataset.report
import deepwit.checkpointing.TensorTreeCheckpointer
import detr.*
import dimwit.*

/** Above which score a relation counts as predicted.
  *
  * A record is a set of relationships, not a ranking of them, so a relation has to be either
  * claimed or not before the graph can be compared with the one the drawing holds. The scene
  * graph literature avoids the question by reporting `R@k` at a fixed `k`; a record cannot,
  * since predicting too many relationships has to cost something.
  */
private val RelationThreshold = 0.5f

/** Scores the l-shape run: `sbt "egtr/runMain egtr.egtrLShapeEval"`. */
@main
def egtrLShapeEval(): Unit =
  import egtr.lshape.{CheckpointRoot, TrainState}
  dimwit.initialize()

  val checkpoints = TensorTreeCheckpointer.latestIn(CheckpointRoot).getOrElse(sys.error(s"no training run in $CheckpointRoot"))
  println(s"reading ${checkpoints.rootPath}")
  val model = EGTR(checkpoints.loadLatest[TrainState].getOrElse(sys.error(s"no checkpoint in ${checkpoints.rootPath}")).params)
  egtrEval(Corpus.LShape, model)

/** Scores the rectilinear run: `sbt "egtr/runMain egtr.egtrRectilinear6to18Eval"`. */
@main
def egtrRectilinear6to18Eval(): Unit =
  import egtr.rectilinear6to18.{CheckpointRoot, TrainState}
  dimwit.initialize()

  val checkpoints = TensorTreeCheckpointer.latestIn(CheckpointRoot).getOrElse(sys.error(s"no training run in $CheckpointRoot"))
  println(s"reading ${checkpoints.rootPath}")
  val model = EGTR(checkpoints.loadLatest[TrainState].getOrElse(sys.error(s"no checkpoint in ${checkpoints.rootPath}")).params)
  egtrEval(Corpus.Rectilinear6to18, model)

/** Scores a trained scene graph model on the whole validation split of a corpus.
  *
  * What the model predicts is read back into the record it stands for and compared with the
  * record the drawing was rendered from, by [[RecordScoring]] — the same comparison the
  * transcription model and the detector are held to, through the same code. Transcribing a
  * document into a graph and detecting objects and relating them are the same task, so they are
  * scored the same way: nodes matched by what they say rather than by the slot they sit in,
  * relationships compared by the nodes they name.
  *
  * The queries a scene graph model works in do not survive that reading: a query holding no
  * object is no node, and the relations it carries relate nothing.
  */
private def egtrEval(corpus: Corpus, model: EGTR[Float32]): Unit =
  val data = DrawingDataset.open(corpus)(Axis[Width], Axis[Height], Axis[Channel], Axis[BoundingBox], Axis[Relationship])(Split.Validation)

  /** The graph the model settles on, holding the relationships it actually claims: the whole of
    * the prediction, on the device, so that only the record comes back.
    */
  val predict = jit: (image: Tensor3[Width, Height, Channel, Float32]) =>
    val graph = model(image)
    val claimed = graph.relations > Tensor.like(graph.relations).fill(RelationThreshold)
    Objects(graph.objects, claimed.asFloat(VType[Float32]))

  val drawings = data
    .samples
    .map(sample => (RecordGraph.of(sample.target), RecordGraph.of(predict(sample.image))))
    .toSeq

  val rightLength = drawings.count((target, predicted) => predicted.size == target.size)
  report("", "records the right length", rightLength, drawings.length)

  Tolerances.foreach: tolerance =>
    RecordScoring.reportAt(tolerance, drawings.map((target, predicted) => RecordScoring.score(target, predicted, tolerance / Canvas)))
