package egtr

import dataset.Box
import dataset.Detection
import dataset.ObjectClass
import dataset.RelationClass
import dataset.RelationClasses
import deepwit.activation.sigmoid
import detr.*
import dimwit.*
import dimwit.optimizer.Adam
import munit.FunSuite

/** The model on a drawing far smaller than the dataset's, so that a whole training run of it
  * fits in a test: three objects, two of them related, in a 32×32 image.
  */
class EGTRSuite extends FunSuite:

  override def beforeAll(): Unit = dimwit.initialize()

  /** The last test trains, which is hundreds of steps and a compilation. */
  override def munitTimeout: scala.concurrent.duration.Duration = scala.concurrent.duration.Duration(5, "min")

  private val imageSize = 32
  private val patchSize = 16
  private val queries = 8
  private val slots = 3
  private val predicates = RelationClass.values.length

  private val detector = DETR.Params.init(
    numLayers = 2,
    numHeads = 2,
    embedding = 16,
    numQueries = queries,
    patchSize = patchSize,
    key = Random.Key(0)
  )
  private val params = EGTR.Params.init(detector, sourceExtent = 16, hiddenExtent = 16, key = Random.Key(1))
  private val model = EGTR(params)
  private val loss = EGTRLoss(VType[Float32], HungarianLoss(VType[Float32])())()

  /** A drawing with two part lines meeting in a corner and a text annotating the first. */
  private val target = SceneGraph[Float32](
    objects = Detection(
      box = Box(
        centerX = vector(Array(0.25f, 0.5f, 0.25f)),
        centerY = vector(Array(0.5f, 0.75f, 0.2f)),
        width = vector(Array(0.5f, 0.1f, 0.1f)),
        height = vector(Array(0.1f, 0.5f, 0.1f))
      ),
      label = Tensor1(Axis[BoundingBox], VType[Int32]).fromArray(
        Array(ObjectClass.PartLine.id, ObjectClass.PartLine.id, ObjectClass.Text.id)
      )
    ),
    relations = edges(
      (0, 1, RelationClass.Connected),
      (1, 0, RelationClass.Connected),
      (2, 0, RelationClass.Annotates)
    )
  )

  private val image = Tensor3(Axis[Width], Axis[Height], Axis[Channel], VType[Float32])
    .fromArray(Array.tabulate(imageSize, imageSize, 1)((x, y, _) => if x == y then 0f else 1f))

  private def vector(values: Array[Float]): Tensor1[BoundingBox, Float32] =
    Tensor1(Axis[BoundingBox], VType[Float32]).fromArray(values)

  private def edges(present: (Int, Int, RelationClass)*): Tensor3[BoundingBox, RelatedBox, RelationClasses, Float32] =
    val dense = Array.fill(slots, slots, predicates)(0f)
    present.foreach((subject, obj, relation) => dense(subject)(obj)(relation.id) = 1f)
    Tensor3(Axis[BoundingBox], Axis[RelatedBox], Axis[RelationClasses], VType[Float32]).fromArray(dense)

  test("scores every ordered pair of queries for every relation"):
    val prediction = model.logits(image)
    assertEquals(prediction.graph.relationLogits.shape.dimensions.toSeq, Seq(queries, queries, predicates))
    assertEquals(prediction.graph.connectivityLogits.shape.dimensions.toSeq, Seq(queries, queries))
    assertEquals(prediction.detection.classLogits.shape(Axis[BoundingBox]), queries)

  test("relates no query to itself, and scores no more than a probability"):
    val decided = model(image)
    val relations = decided.relations.toArray
    for query <- 0 until queries; relation <- 0 until predicates do
      assertEquals(relations(query)(query)(relation), 0f, s"query $query relates to itself")
    assert(relations.flatten.flatten.forall(score => score >= 0f && score <= 1f))

  test("the target graph keeps its edges when the matching lines it up with the queries"):
    // Every target object is matched to a distinct query, so permuting the graph onto the queries
    // moves the edges around but neither drops nor duplicates any.
    val matched = loss.detection.matched(model.logits(image).detection, target.objects)
    val aligned = loss.targetGraph(matched, target.relations)
    assertEquals(aligned.shape.dimensions.toSeq, Seq(queries, queries, predicates))
    assertEquals(aligned.sum.item, target.relations.sum.item)
    // …and onto the queries the target objects were assigned, specifically.
    val slotOf = matched.slot.toArray
    val edges = aligned.toArray
    for
      subject <- 0 until queries
      obj <- 0 until queries
      relation <- 0 until predicates
      if edges(subject)(obj)(relation) > 0f
    do
      assertEquals(
        target.relations.toArray(slotOf(subject))(slotOf(obj))(relation),
        1f,
        s"query pair ($subject, $obj) carries an edge its target slots do not"
      )

  test("training on one drawing learns both its objects and its edges"):
    val optimizer = Adam(learningRate = Tensor0(3e-3f))
    val step = jit: (params: EGTR.Params[Float32], state: dimwit.optimizer.AdamState[EGTR.Params[Float32]]) =>
      val (cost, gradients) = Autodiff.valueAndGrad((p: EGTR.Params[Float32]) => loss(EGTR(p).logits(image), target))(params)
      val (next, nextState) = optimizer.update(gradients, params, state)
      (cost, next, nextState)

    /* The drawing is a diagonal line with no objects in it, so the detection never really fits
     * and the relation labels stay damped by the uncertainty of eq. 8 — the edges separate from
     * the rest by two orders of magnitude, but at low probabilities throughout. That is the
     * curriculum working, so what is asserted below is the separation and not the confidence. */
    val (first, trained) = (1 to 600).foldLeft((Option.empty[Float], (params, optimizer.init(params)))):
      case ((first, (params, state)), _) =>
        val (cost, next, nextState) = step(params, state)
        (first.orElse(Some(cost.item)), (next, nextState))
    val (finalParams, _) = trained
    val last = loss(EGTR(finalParams).logits(image), target).item

    assert(last < first.get * 0.5f, s"the loss barely moved: ${first.get} -> $last")

    // Overfitted on one drawing, the graph it predicts should be the one it was given.
    val trainedModel = EGTR(finalParams)
    val matched = loss.detection.matched(trainedModel.logits(image).detection, target.objects)
    val aligned = loss.targetGraph(matched, target.relations).toArray
    val predicted = sigmoid(trainedModel.logits(image).graph.relationLogits).toArray
    val edges = for
      subject <- 0 until queries
      obj <- 0 until queries
      relation <- 0 until predicates
    yield (aligned(subject)(obj)(relation) > 0f, predicted(subject)(obj)(relation))
    val (present, absent) = edges.partition(_._1)
    assert(present.nonEmpty)
    assert(
      present.map(_._2).min > absent.map(_._2).max,
      s"the edges are not separated from the rest: present ${present.map(_._2).min}, absent ${absent.map(_._2).max}"
    )
