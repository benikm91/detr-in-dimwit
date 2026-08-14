import dataset.ObjectClass
import deepwit.base.AffineLayer
import deepwit.base.relu
import deepwit.base.sigmoid
import deepwit.embedder.ImageToPatchEmbedder
import deepwit.normalization.LayerNorm
import deepwit.transformer.CrossTransformer
import deepwit.transformer.CrossTransformerLayer
import deepwit.transformer.MLPEmbeddingMixer
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
case class DETR[V: IsFloating](params: DETR.Params[V]) extends (Tensor3[Width, Height, Channel, V] => DETR.Prediction[V]):

  import DETR.BoxCoordinate
  import DETR.Embedding
  import DETR.Patch
  import DETR.Prediction

  private val patches = ImageToPatchEmbedder(params.patchEmbedder)
  private val encoder = Transformer.bidirectional(Axis[Patch], params.encoder)
  private val decoder = CrossTransformer(
    params.decoder.map(new CrossTransformerLayer(_, fullMask[BoundingBox, Patch], fullMask[BoundingBox, BoundingBox])),
    LayerNorm(params.decoderNorm)
  )
  private val classify = AffineLayer(params.classification)
  private val boxHidden1 = AffineLayer(params.boxHidden1)
  private val boxHidden2 = AffineLayer(params.boxHidden2)
  private val boxOutput = AffineLayer(params.boxOutput)

  override def apply(image: Tensor3[Width, Height, Channel, V]): Prediction[V] =
    val objects = decoder(encoder(patches(image)), params.objectQueries)
    val boxes = objects.vmap(Axis[BoundingBox])(box)
    Prediction(
      centerX = boxes.slice(Axis[BoxCoordinate].at(0)),
      centerY = boxes.slice(Axis[BoxCoordinate].at(1)),
      width = boxes.slice(Axis[BoxCoordinate].at(2)),
      height = boxes.slice(Axis[BoxCoordinate].at(3)),
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

  case class Prediction[V](
      centerX: Tensor1[BoundingBox, V],
      centerY: Tensor1[BoundingBox, V],
      width: Tensor1[BoundingBox, V],
      height: Tensor1[BoundingBox, V],
      classLogits: Tensor2[BoundingBox, ObjectClasses, V]
  ):
    /** `vmap` cannot return a case class, so batching goes through the plain tuple. */
    def toTuple: (
        Tensor1[BoundingBox, V],
        Tensor1[BoundingBox, V],
        Tensor1[BoundingBox, V],
        Tensor1[BoundingBox, V],
        Tensor2[BoundingBox, ObjectClasses, V]
    ) = (centerX, centerY, width, height, classLogits)

  case class PredictionBatch[S, V](
      centerX: Tensor2[S, BoundingBox, V],
      centerY: Tensor2[S, BoundingBox, V],
      width: Tensor2[S, BoundingBox, V],
      height: Tensor2[S, BoundingBox, V],
      classLogits: Tensor3[S, BoundingBox, ObjectClasses, V]
  ):
    def at(sample: Int)(using Label[S]): Prediction[V] =
      Prediction(
        centerX = centerX.slice(Axis[S].at(sample)),
        centerY = centerY.slice(Axis[S].at(sample)),
        width = width.slice(Axis[S].at(sample)),
        height = height.slice(Axis[S].at(sample)),
        classLogits = classLogits.slice(Axis[S].at(sample))
      )

  object PredictionBatch:
    def apply[S, V](
        batched: (
            Tensor2[S, BoundingBox, V],
            Tensor2[S, BoundingBox, V],
            Tensor2[S, BoundingBox, V],
            Tensor2[S, BoundingBox, V],
            Tensor3[S, BoundingBox, ObjectClasses, V]
        )
    ): PredictionBatch[S, V] =
      PredictionBatch(batched._1, batched._2, batched._3, batched._4, batched._5)

  case class Params[V](
      patchEmbedder: ImageToPatchEmbedder.Params[Width, Height, Channel, Embedding, V],
      encoder: Transformer.Params[Embedding, V],
      decoder: List[CrossTransformerLayer.Params[Embedding, Embedding, V]],
      decoderNorm: LayerNorm.Params[Embedding, V],
      objectQueries: Tensor2[BoundingBox, Embedding, V],
      classification: AffineLayer.Params[Embedding, ObjectClasses, V],
      boxHidden1: AffineLayer.Params[Embedding, BoxHidden, V],
      boxHidden2: AffineLayer.Params[BoxHidden, Prime[BoxHidden], V],
      boxOutput: AffineLayer.Params[Prime[BoxHidden], BoxCoordinate, V]
  )

  object Params:

    given tensorTree: TensorTree[Params[Float32]] = TensorTree.derived
    given tree: TreeOf[Params[Float32], Float32] = TreeOf.derived

    def init[V: IsFloating](numEncoderLayers: Int, numDecoderLayers: Int)(
        patchWidthExtent: AxisExtent[Width],
        patchHeightExtent: AxisExtent[Height],
        channelExtent: AxisExtent[Channel],
        boundingBoxExtent: AxisExtent[BoundingBox],
        headExtent: AxisExtent[Head],
        headQueryExtent: AxisExtent[HeadQuery],
        headKeyExtent: AxisExtent[HeadKey],
        headValueExtent: AxisExtent[HeadValue],
        embeddingExtent: AxisExtent[Embedding],
        embeddingMixedExtent: AxisExtent[MLPEmbeddingMixer.EmbeddingMixed],
        boxHiddenExtent: AxisExtent[BoxHidden],
        vtype: VType[V],
        key: Random.Key
    ): Params[V] =
      val (patchKey, encoderKey, decoderKey, queryKey, headsKey) = key.splitToTuple(5)
      val (classKey, box1Key, box2Key, box3Key) = headsKey.splitToTuple(4)
      val boxHiddenExtent2 = Axis[Prime[BoxHidden]] -> boxHiddenExtent.size
      Params(
        patchEmbedder = ImageToPatchEmbedder.Params.xavierUniform(
          patchWidthExtent,
          patchHeightExtent,
          channelExtent,
          embeddingExtent,
          vtype,
          patchKey
        ),
        encoder = Transformer.Params.xavierUniformDepthScaled(
          numEncoderLayers,
          headExtent,
          headQueryExtent,
          headKeyExtent,
          headValueExtent,
          embeddingExtent,
          embeddingMixedExtent,
          vtype,
          encoderKey
        ),
        decoder = decoderKey
          .split(numDecoderLayers)
          .map: layerKey =>
            CrossTransformerLayer.Params.xavierUniformDepthScaled(numDecoderLayers)(
              headExtent,
              headQueryExtent,
              headKeyExtent,
              headValueExtent,
              embeddingExtent,
              embeddingExtent,
              embeddingMixedExtent,
              vtype,
              layerKey
            )
          .toList,
        decoderNorm = LayerNorm.Params.identity(embeddingExtent, vtype),
        objectQueries = deepwit.init.xavierUniform(boundingBoxExtent, embeddingExtent, vtype, queryKey),
        classification = AffineLayer.Params.xavierUniform(embeddingExtent, Axis[ObjectClasses] -> ObjectClass.values.length, vtype, classKey),
        boxHidden1 = AffineLayer.Params.xavierUniform(embeddingExtent, boxHiddenExtent, vtype, box1Key),
        boxHidden2 = AffineLayer.Params.xavierUniform(boxHiddenExtent, boxHiddenExtent2, vtype, box2Key),
        boxOutput = AffineLayer.Params.xavierUniform(boxHiddenExtent2, Axis[BoxCoordinate] -> 4, vtype, box3Key)
      )
