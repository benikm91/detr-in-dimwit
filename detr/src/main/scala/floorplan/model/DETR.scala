package detr.floorplan.model

import detr.*
import detr.model.*
import deepwit.base.AffineLayer
import documentEncoder.DocumentEncoder
import deepwit.normalization.LayerNorm
import deepwit.init.Init
import dimwit.*
import syntheticFloorplansDataset.FloorplanClass
import syntheticFloorplansDataset.FloorplanClasses
import syntheticFloorplansDataset.FloorplanNodes

/** DETR, [[https://arxiv.org/abs/2005.12872 End-to-End Object Detection with Transformers]],
  * with a vision transformer in place of the convolutional backbone — see `README.md` for
  * the divergences from the paper.
  *
  * The image is embedded patch by patch and attended over by the encoder. The decoder turns
  * a fixed set of learned object queries into one embedding per [[Query]], from which the
  * [[NodeHead]] predicts a node of the floor plan, or none.
  */
class DETR[V: IsFloating](params: DETR.Params[V]) extends (Tensor3[Width, Height, Channel, V] => FloorplanNodes[Query]):

  private val encodeDocument = DocumentEncoder(params.encoder)
  private val decoder = DETRDecoder(Axis[Patch], Axis[Query], params.decoder)
  private val nodeHead = NodeHead(params.nodeHead)

  /** The node every query answers with, [[FloorplanClass.NoNode]] where it answers with none. */
  override def apply(image: Tensor3[Width, Height, Channel, V]): FloorplanNodes[Query] =
    nodeHead.decide(logits(image))

  def logits(image: Tensor3[Width, Height, Channel, V]): NodeHead.NodeLogits[V] =
    nodeHead(decoder(encodeDocument(image), params.objectQueries))

object DETR:

  trait Embedding derives Label

  case class Params[V](
      encoder: DocumentEncoder.Params[Embedding, V],
      decoder: DETRDecoder.Params[Embedding, Embedding, V],
      objectQueries: Tensor2[Query, Embedding, V],
      nodeHead: NodeHead.Params[V]
  )

  object Params:

    given tensorTree: TensorTree[Params[Float32]] = TensorTree.derived
    given tree: TreeOf[Params[Float32], Float32] = TreeOf.derived

    def init(
        numLayers: Int,
        numHeads: Int,
        embedding: Int,
        numQueries: Int,
        canvas: Int,
        key: Key
    ) =
      val (encoderKey, decoderKey, queryKey, headsKey) = key.splitToTuple(4)
      val (classKey, startXKey, startYKey, endXKey, endYKey) = headsKey.splitToTuple(5)
      val queryExtent = Axis[Query] -> numQueries
      val embeddingExtent = Axis[DETR.Embedding] -> embedding
      val embeddingMixedExtent = Axis[EmbeddingMixed] -> embeddingExtent.size * 4
      val pixelExtent = Axis[Pixel] -> canvas
      Params(
        encoder = DocumentEncoder.Params.xavierUniformDepthScaled(
          numLayers,
          numHeads,
          embeddingExtent,
          Axis[DocumentEncoder.EmbeddingMixed] -> embeddingMixedExtent.size,
          encoderKey
        ),
        decoder = DETRDecoder.Params.xavierUniformDepthScaled(
          numLayers,
          numHeads,
          embeddingExtent,
          embeddingExtent,
          embeddingMixedExtent,
          decoderKey
        ),
        objectQueries = Init.xavierUniform(queryExtent, embeddingExtent, queryKey),
        nodeHead = NodeHead.Params(
          nodeClass = AffineLayer.Params.init(embeddingExtent, Axis[FloorplanClasses] -> FloorplanClass.values.length, classKey),
          startX = AffineLayer.Params.init(embeddingExtent, pixelExtent, startXKey),
          startY = AffineLayer.Params.init(embeddingExtent, pixelExtent, startYKey),
          endX = AffineLayer.Params.init(embeddingExtent, pixelExtent, endXKey),
          endY = AffineLayer.Params.init(embeddingExtent, pixelExtent, endYKey)
        )
      )
