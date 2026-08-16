import dataset.Box
import dataset.Detection
import dataset.ObjectClass
import deepwit.activation.relu
import deepwit.activation.sigmoid
import deepwit.base.AffineLayer
import deepwit.embedder.ImageToPatchEmbedder
import deepwit.normalization.LayerNorm
import deepwit.transformer.CrossTransformer
import deepwit.transformer.CrossTransformerLayer
import deepwit.transformer.EmbeddingMixed
import deepwit.transformer.Transformer
import deepwit.transformer.attention.Head
import deepwit.transformer.attention.HeadKey
import deepwit.transformer.attention.HeadQuery
import deepwit.transformer.attention.HeadValue
import deepwit.transformer.fullMask
import dimwit.*
import dimwit.tensor.Tensor4

/** DETR, [[https://arxiv.org/abs/2005.12872 End-to-End Object Detection with Transformers]],
  * with a vision transformer in place of the convolutional backbone — see `README.md` for
  * the divergences from the paper.
  *
  * The image is embedded patch by patch and attended over by the encoder. The decoder turns
  * a fixed set of learned object queries into one embedding per [[BoundingBox]] slot, from
  * which a linear layer predicts the class and a three layer perceptron the box.
  */
class DETR[V: IsFloating](params: DETR.Params[V]) extends (Tensor3[Width, Height, Channel, V] => ObjectDetection[V]):

  import DETR.BoxCoordinate
  import DETR.Embedding
  import DETR.Patch
  import DETR.Prediction

  private val patches = ImageToPatchEmbedder(params.patchEmbedder)
  private val encoder = Transformer.bidirectional(Axis[Patch], params.encoder)
  private val decoder = CrossTransformer.bidirectional(Axis[Patch], Axis[BoundingBox], params.decoder)
  private val classify = AffineLayer(params.classification)
  private val boxHidden1 = AffineLayer(params.boxHidden1)
  private val boxHidden2 = AffineLayer(params.boxHidden2)
  private val boxOutput = AffineLayer(params.boxOutput)

  /** The detected objects: per query the box and its most likely class. */
  override def apply(image: Tensor3[Width, Height, Channel, V]): ObjectDetection[V] =
    val prediction = logits(image)
    Detection(prediction.box, prediction.classLogits.argmax(Axis[ObjectClasses]))

  /** What the model scores before deciding: a box and unnormalized class scores per query. */
  def logits(image: Tensor3[Width, Height, Channel, V]): Prediction[V] =
    val objects = decoder(encoder(patches(image)), params.objectQueries)
    val boxes = objects.vmap(Axis[BoundingBox])(box)
    Prediction(
      box = Box(
        centerX = boxes.slice(Axis[BoxCoordinate].at(0)),
        centerY = boxes.slice(Axis[BoxCoordinate].at(1)),
        width = boxes.slice(Axis[BoxCoordinate].at(2)),
        height = boxes.slice(Axis[BoxCoordinate].at(3))
      ),
      classLogits = objects.vmap(Axis[BoundingBox])(classify)
    )

  private def box(embedding: Tensor1[Embedding, V]): Tensor1[BoxCoordinate, V] =
    sigmoid(boxOutput(relu(boxHidden2(relu(boxHidden1(embedding))))))

object DETR:

  trait Embedding derives Label
  trait BoxHidden derives Label
  trait BoxCoordinate derives Label

  /** The image flattened into the encoder's sequence of patches. */
  type Patch = Width |*| Height

  /** What the model scores: a [[Box]] and the class scores of every query. */
  case class Prediction[V](
      box: Box[Tuple1[BoundingBox], V],
      classLogits: Tensor2[BoundingBox, ObjectClasses, V]
  )

  case class Params[V](
      patchEmbedder: ImageToPatchEmbedder.Params[Width, Height, Channel, Embedding, V],
      encoder: Transformer.Params[Embedding, V],
      decoder: CrossTransformer.Params[Embedding, Embedding, V],
      objectQueries: Tensor2[BoundingBox, Embedding, V],
      classification: AffineLayer.Params[Embedding, ObjectClasses, V],
      boxHidden1: AffineLayer.Params[Embedding, BoxHidden, V],
      boxHidden2: AffineLayer.Params[BoxHidden, Prime[BoxHidden], V],
      boxOutput: AffineLayer.Params[Prime[BoxHidden], BoxCoordinate, V]
  )

  object Params:

    given tensorTree: TensorTree[Params[Float32]] = TensorTree.derived
    given tree: TreeOf[Params[Float32], Float32] = TreeOf.derived

    def init(
        numLayers: Int,
        numHeads: Int,
        embedding: Int,
        numQueries: Int,
        key: Key
    ) =
      val (patchKey, encoderKey, decoderKey, queryKey, headsKey) = key.splitToTuple(5)
      val (classKey, box1Key, box2Key, box3Key) = headsKey.splitToTuple(4)
      val boundingBoxExtent = Axis[BoundingBox] -> numQueries
      val embeddingExtent = Axis[DETR.Embedding] -> embedding
      val embeddingMixedExtent = Axis[EmbeddingMixed] -> embeddingExtent.size * 4
      val boxHiddenExtent = Axis[BoxHidden] -> embedding
      val boxHiddenExtent2 = Axis[Prime[BoxHidden]] -> embedding
      Params(
        patchEmbedder = ImageToPatchEmbedder.Params.xavierUniform(
          Axis[Width] -> 16,
          Axis[Height] -> 16,
          Axis[Channel] -> 1,
          embeddingExtent,
          VType[Float32],
          patchKey
        ),
        encoder = Transformer.Params.xavierUniformDepthScaled(
          numLayers,
          numHeads,
          embeddingExtent,
          embeddingMixedExtent,
          VType[Float32],
          encoderKey
        ),
        decoder = CrossTransformer.Params.xavierUniformDepthScaled(
          numLayers,
          numHeads,
          embeddingExtent,
          embeddingExtent,
          embeddingMixedExtent,
          VType[Float32],
          decoderKey
        ),
        objectQueries = deepwit.init.xavierUniform(boundingBoxExtent, embeddingExtent, VType[Float32], queryKey),
        classification = AffineLayer.Params.xavierUniform(embeddingExtent, Axis[ObjectClasses] -> ObjectClass.values.length, VType[Float32], classKey),
        boxHidden1 = AffineLayer.Params.xavierUniform(embeddingExtent, boxHiddenExtent, VType[Float32], box1Key),
        boxHidden2 = AffineLayer.Params.xavierUniform(boxHiddenExtent, boxHiddenExtent2, VType[Float32], box2Key),
        boxOutput = AffineLayer.Params.xavierUniform(boxHiddenExtent2, Axis[BoxCoordinate] -> 4, VType[Float32], box3Key)
      )
