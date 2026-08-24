import dataset.LShapeDataset
import dataset.LShapeDataset.Split
import dataset.RelationClass
import dataset.RelationClasses
import dataset.Tolerances
import dataset.at
import dataset.report
import dimwit.*

import DetectionScoring.Slot
import DetectionScoring.isDetected
import DetectionScoring.slots

/** How far down the ranking of a drawing's triplets recall is measured, the `R@k` of the scene
  * graph literature.
  *
  * A graph of 32 queries over 2 relations has 2048 entries of which a drawing fills some 14, and
  * ranking those entries is what the paper scores rather than deciding them at a cut-off, so it
  * is what most of this reports too.
  */
private val Ranks = Seq(20, 50)

/** Above which triplet score a relation counts as predicted, for the one line that reports a
  * decision rather than a ranking. It is the question a consumer of the graph really has, and it
  * asks the scores to be calibrated where the ranking only asks them to be ordered.
  */
private val RelationThreshold = 0.5f

/** Scores a trained scene graph model on the whole validation split:
  * `sbt "egtr/runMain egtrEval"`.
  *
  * Nodes are matched and scored exactly as [[detrEval]] scores them, so the object lines are
  * comparable between a detector and a scene graph model. The edges are then scored through that
  * same matching, on top of the nodes rather than beside them — a target edge is only credited
  * when both of its objects were also detected, since an edge between misplaced boxes relates
  * nothing:
  *
  *   - `relations R@k` — of the target edges, how many are among the `k` best scoring triplets
  *     of their drawing. `connected` and `annotates` break the same ranking down per relation,
  *     which matters because the two are not equally frequent.
  *   - `drawings fully correct` — every query slot right and every target edge outranking every
  *     triplet that is not one: the whole graph of the drawing, in one number, without a
  *     threshold to pick.
  *   - `relations over 0.5` — of the target edges, how many actually score above
  *     [[RelationThreshold]]. Unlike the lines above this one asks the scores to be calibrated
  *     and not merely ordered.
  */
@main
def egtrEval(run: String*): Unit =
  dimwit.initialize()

  val checkpoints = checkpointsIn(EGTRCheckpointRoot, run)
  println(s"reading ${checkpoints.rootPath}")
  val model = EGTR(checkpoints.loadLatest[EGTRTrainState].getOrElse(sys.error(s"no checkpoint in ${checkpoints.rootPath}")).params)
  val loss = EGTRLoss(VType[Float32], HungarianLoss(VType[Float32])())()
  val data = LShapeDataset.open(Axis[Width], Axis[Height], Axis[Channel], Axis[BoundingBox])(Split.Validation)

  /* Matching and scoring stay on the device: only the four arrays a drawing is scored from come
   * back, so the whole split is one compiled function called once per drawing. */
  val evaluate = jit: (image: Tensor3[Width, Height, Channel, Float32], target: SceneGraph[Float32]) =>
    val prediction = model.logits(image)
    val matched = loss.detection.matched(prediction.detection, target.objects)
    val graph = model.decide(prediction)
    Scoring(
      targets = matched.targets,
      detected = graph.objects,
      targetEdges = loss.targetGraph(matched, target.relations),
      predictedEdges = graph.relations
    )

  // The split is predicted once and every tolerance scored off the same matched slots.
  val drawings = data
    .objects()
    .map: sample =>
      val scoring = evaluate(sample.image, SceneGraph(sample.target.detection, sample.target.relations))
      GraphPrediction(
        nodes = slots(scoring.targets, data.imageWidth, data.imageHeight)
          .zip(slots(scoring.detected, data.imageWidth, data.imageHeight)),
        targetEdges = triplets(scoring.targetEdges).filter(_.score > 0.5f).map(_.triplet).toSet,
        scores = triplets(scoring.predictedEdges)
      )
    .toSeq

  // Whether an empty query stays empty is a matter of its class alone, so any tolerance scores it.
  val empty = drawings.flatMap(_.nodes).filter(!_._1.isObject)
  report("", "empty queries kept empty", empty.count(!_._2.isObject), empty.size)

  Tolerances.foreach: tolerance =>
    val scored = drawings.map(_.at(tolerance))
    val nodes = scored.flatMap(_.nodes)
    val objects = nodes.filter(_.target.isObject)
    val claimed = nodes.filter(_.predicted.isObject)
    report(at(tolerance), "objects detected", objects.count(_.isDetected), objects.size)
    report(at(tolerance), "detections correct", claimed.count(_.isDetected), claimed.size)

    val leadingRank = Ranks.head
    RelationClass.values.foreach: relation =>
      report(
        at(tolerance),
        s"${relation.toString.toLowerCase} R@$leadingRank",
        scored.map(_.recalledWithin(leadingRank).count(_.relation == relation)).sum,
        scored.map(_.targetEdges.count(_.relation == relation)).sum
      )
    Ranks.foreach: rank =>
      report(at(tolerance), s"relations R@$rank", scored.map(_.recalledWithin(rank).size).sum, scored.map(_.targetEdges.size).sum)
    report(at(tolerance), "relations over 0.5", scored.map(_.recalledOverThreshold.size).sum, scored.map(_.targetEdges.size).sum)
    report(at(tolerance), "drawings fully correct", scored.count(_.isFullyCorrect), scored.size)

/** What one drawing is scored from, once the device is done with it. */
private case class Scoring(
    targets: ObjectDetection[Float32],
    detected: ObjectDetection[Float32],
    targetEdges: Tensor3[BoundingBox, RelatedBox, RelationClasses, Float32],
    predictedEdges: Tensor3[BoundingBox, RelatedBox, RelationClasses, Float32]
)

/** One edge of a graph: an ordered pair of query slots and the relation between them. */
private case class Triplet(subject: Int, obj: Int, relation: RelationClass)

private case class Scored(triplet: Triplet, score: Float)

/** One drawing as predicted, before the nodes are judged at a tolerance. */
private case class GraphPrediction(nodes: Seq[(Slot, Slot)], targetEdges: Set[Triplet], scores: Seq[Scored]):

  def at(tolerance: Float): ScoredGraph =
    ScoredGraph(
      nodes = nodes.map((target, predicted) => ScoredNode(target, predicted, isDetected(target, predicted, tolerance))),
      targetEdges = targetEdges,
      scores = scores
    )

/** One query slot: what it should hold, what it holds, and whether that counts. */
private case class ScoredNode(target: Slot, predicted: Slot, isDetected: Boolean)

/** One drawing, scored at one tolerance. */
private case class ScoredGraph(nodes: Seq[ScoredNode], targetEdges: Set[Triplet], scores: Seq[Scored]):

  /** Whether both objects an edge relates were detected — without them there is nothing for the
    * edge to hold between, so it cannot be credited.
    */
  private def relatesDetected(triplet: Triplet): Boolean =
    nodes(triplet.subject).isDetected && nodes(triplet.obj).isDetected

  /** The triplets of the drawing, best scoring first. */
  private lazy val ranked: Seq[Triplet] = scores.sortBy(-_.score).map(_.triplet)

  /** Which target edges, between objects that were detected, are among the `rank` best scoring
    * triplets of the drawing.
    */
  def recalledWithin(rank: Int): Seq[Triplet] =
    ranked.take(rank).filter(triplet => targetEdges.contains(triplet) && relatesDetected(triplet))

  /** Which target edges, between objects that were detected, actually score above the threshold. */
  def recalledOverThreshold: Seq[Scored] =
    scores.filter(scored => scored.score >= RelationThreshold && targetEdges.contains(scored.triplet) && relatesDetected(scored.triplet))

  /** Every query slot right, and the target edges the whole top of the ranking: nothing missing,
    * nothing spurious and nothing to threshold.
    */
  def isFullyCorrect: Boolean =
    nodes.forall(_.isDetected) && ranked.take(targetEdges.size).toSet == targetEdges

/** The grid of relation scores as one entry per triplet. */
private def triplets(graph: Tensor3[BoundingBox, RelatedBox, RelationClasses, Float32]): Seq[Scored] =
  val scores = graph.toArray
  for
    subject <- scores.indices
    obj <- scores(subject).indices
    relation <- scores(subject)(obj).indices
  yield Scored(Triplet(subject, obj, RelationClass.fromId(relation)), scores(subject)(obj)(relation))
