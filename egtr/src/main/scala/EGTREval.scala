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
import dimwit.*

/** Scores a trained scene graph model on the whole validation split of the corpus its setup
  * names.
  *
  * What the model predicts is read back into the record it stands for and compared with the
  * record the drawing was rendered from, by [[RecordScoring]] — the same comparison
  * [[d2gEval]] and [[detrEval]] make, through the same code. Transcribing a document into a
  * graph and detecting objects and relating them are the same task, so they are scored the same
  * way: nodes matched by what they say rather than by the slot they sit in, relationships
  * compared by the nodes they name.
  *
  * The queries a scene graph model works in do not survive that reading: a query holding no
  * object is no node, and the relations it carries relate nothing.
  */
def egtrEval(setup: EGTRSetup): Unit =
  dimwit.initialize()

  val checkpoints = TensorTreeCheckpointer.latestIn(setup.checkpointRoot).getOrElse(sys.error(s"no training run in ${setup.checkpointRoot}"))
  println(s"reading ${checkpoints.rootPath}")
  val model = EGTR(checkpoints.loadLatest[EGTRTrainState].getOrElse(sys.error(s"no checkpoint in ${checkpoints.rootPath}")).params)
  val data = DrawingDataset.open(setup.corpus)(Axis[Width], Axis[Height], Axis[Channel], Axis[BoundingBox], Axis[Relationship])(Split.Validation)

  /** The graph the model settles on, holding the relationships it actually claims: the whole of
    * the prediction, on the device, so that only the record comes back.
    */
  val predict = jit: (image: Tensor3[Width, Height, Channel, Float32]) =>
    val graph = model(image)
    val claimed = graph.relations > Tensor.like(graph.relations).fill(setup.relationThreshold)
    Objects(graph.objects, claimed.asFloat(VType[Float32]))

  val drawings = data
    .samples
    .map(sample => (RecordGraph.of(sample.target), RecordGraph.of(predict(sample.image))))
    .toSeq

  val rightLength = drawings.count((target, predicted) => predicted.size == target.size)
  report("", "records the right length", rightLength, drawings.length)

  Tolerances.foreach: tolerance =>
    RecordScoring.reportAt(tolerance, drawings.map((target, predicted) => RecordScoring.score(target, predicted, tolerance / Canvas)))
