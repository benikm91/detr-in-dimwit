import dataset.Detection
import dataset.ObjectClass
import dataset.RelationClass
import dataset.RelationClasses
import deepwit.activation.sigmoid
import deepwit.activation.softmax
import dimwit.*
import dimwit.Conversions.given

import scala.language.implicitConversions

/** EGTR, [[https://arxiv.org/abs/2404.02072 Extracting Graph from Transformer for Scene Graph
  * Generation]], on top of the [[DETR]] of this repository — see `README.md` for the divergences
  * from the paper.
  *
  * A scene graph is a detection plus the edges between what was detected. EGTR predicts both at
  * once from one detector: the object queries of the decoder are the nodes, and the edges are
  * read out of the self-attention that those queries already run onto each other, by the shallow
  * [[RelationExtractor]]. There are no relation queries, no second decoder and no separately
  * trained detector.
  *
  * On the l-shape drawings the nodes are the part lines and the dimension annotations, and the
  * edges are which lines meet in a corner ([[RelationClass.Connected]]) and which line each
  * annotation measures ([[RelationClass.Annotates]]).
  */
class EGTR[V: IsFloating](params: EGTR.Params[V]) extends (Tensor3[Width, Height, Channel, V] => SceneGraph[V]):

  import EGTR.Prediction

  private val detector = DETR(params.detector)
  private val extractor = RelationExtractor(params.relations)

  /** The scene graph the model settles on: see [[decide]]. */
  override def apply(image: Tensor3[Width, Height, Channel, V]): SceneGraph[V] =
    decide(logits(image))

  /** What the model scores before deciding, i.e. what the loss works on. */
  def logits(image: Tensor3[Width, Height, Channel, V]): Prediction[V] =
    val decoded = detector.decode(image)
    Prediction(detector.predict(decoded.objects), extractor(decoded))

  /** The scores of §3.3.3, turned into a graph over the detected objects.
    *
    * A triplet is only as good as the two objects it relates, so the score of a relation is
    * multiplied by the class probability of its subject and of its object, and by the
    * probability that the pair is connected at all. Nothing relates to itself, so the diagonal
    * is cleared.
    *
    * That class probability is taken over the object classes only, as DETR scores a detection:
    * a query saying there is nothing in its slot says so with a high probability, and reading
    * that as confidence would let the queries holding nothing — most of them — carry the best
    * scoring relations in the drawing.
    */
  def decide(prediction: Prediction[V]): SceneGraph[V] =
    val probability = prediction.detection.classLogits.vapply(Axis[ObjectClasses])(softmax)
    val queries = probability.shape.extent(Axis[BoundingBox])
    val triplets = Shape3(
      queries,
      Axis[RelatedBox] -> queries.size,
      Axis[RelationClasses] -> RelationClass.values.length
    )
    val isObject = Tensor1(Axis[ObjectClasses], VType[V])
      .fromArray(ObjectClass.values.map(objectClass => if objectClass == ObjectClass.NoObject then 0f else 1f))
    val confidence = (probability * isObject.broadcastTo(probability.shape)).max(Axis[ObjectClasses])
    SceneGraph(
      objects = Detection(prediction.detection.box, probability.argmax(Axis[ObjectClasses])),
      relations = sigmoid(prediction.graph.relationLogits)
        * confidence.broadcastTo(triplets)
        * confidence.relabelTo(Axis[RelatedBox]).broadcastTo(triplets)
        * sigmoid(prediction.graph.connectivityLogits).broadcastTo(triplets)
        * (1f -! Tensor2.eye(queries, VType[V])).broadcastTo(triplets)
    )

object EGTR:

  /** What the model scores: a detection per query and a graph over the pairs of queries. */
  case class Prediction[V](
      detection: DETR.Prediction[V],
      graph: RelationExtractor.Graph[V]
  )

  case class Params[V](
      detector: DETR.Params[V],
      relations: RelationExtractor.Params[V]
  )

  object Params:

    given tensorTree: TensorTree[Params[Float32]] = TensorTree.derived
    given tree: TreeOf[Params[Float32], Float32] = TreeOf.derived

    /** Wraps a detector in a relation extractor sized to it.
      *
      * The detector is a parameter rather than a set of sizes so that a scene graph model can be
      * started from a trained one — which is how the paper trains it, the detection being the
      * harder half of the task and the relations being read off the detector's own attention.
      *
      * @param sourceExtent The width one source projects a query into, `d_model` in the paper.
      * @param hiddenExtent The width of the hidden layers of the relation and connectivity heads.
      */
    def init(
        detector: DETR.Params[Float32],
        sourceExtent: Int,
        hiddenExtent: Int,
        key: Key
    ): Params[Float32] =
      Params(
        detector = detector,
        relations = RelationExtractor.Params.xavierUniform(
          detector,
          Axis[RelationSource] -> sourceExtent,
          Axis[RelationHidden] -> hiddenExtent,
          VType[Float32],
          key
        )
      )
