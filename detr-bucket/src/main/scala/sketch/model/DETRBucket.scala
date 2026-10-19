package detrbucket.sketch.model

import detr.*
import detr.model.*
import detrbucket.sketch.Buckets
import dataset.IsConstruction
import dataset.RecordNodes
import deepwit.base.AffineLayer
import documentEncoder.DocumentEncoder
import deepwit.normalization.LayerNorm
import deepwit.init.Init
import dimwit.*

/** DETR with its object queries split by class: [[detr.sketch.model.DETR]]'s encoder and decoder,
  * but every query belongs to a bucket of [[Buckets]] and answers only with a node of that bucket's
  * class, or with none.
  *
  * The image is embedded patch by patch and attended over by the encoder. The decoder turns the
  * learned object queries, every bucket's one after the other, into one embedding per [[Query]],
  * from which the [[BucketHead]] predicts whether the query answers, and where.
  */
class DETRBucket[V: IsFloating](params: DETRBucket.Params[V], buckets: Buckets) extends (Tensor3[Width, Height, Channel, V] => RecordNodes[Query]):

  private val encodeDocument = DocumentEncoder(params.encoder)
  private val decoder = DETRDecoder(Axis[Patch], Axis[Query], params.decoder)
  private val nodeHead = BucketHead(params.nodeHead, buckets)

  /** The node every query answers with, [[dataset.NodeClass.NoNode]] where it answers with none. */
  override def apply(image: Tensor3[Width, Height, Channel, V]): RecordNodes[Query] =
    nodeHead.decide(logits(image))

  def logits(image: Tensor3[Width, Height, Channel, V]): BucketHead.NodeLogits[V] =
    nodeHead(decoder(encodeDocument(image), params.objectQueries))

object DETRBucket:

  trait Embedding derives Label

  case class Params[V](
      encoder: DocumentEncoder.Params[Embedding, V],
      decoder: DETRDecoder.Params[Embedding, Embedding, V],
      objectQueries: Tensor2[Query, Embedding, V],
      nodeHead: BucketHead.Params[V]
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
      val (presenceKey, constructionKey, startXKey, startYKey, endXKey, endYKey, midXKey, midYKey) = headsKey.splitToTuple(8)
      val queryExtent = Axis[Query] -> numQueries
      val embeddingExtent = Axis[DETRBucket.Embedding] -> embedding
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
        nodeHead = BucketHead.Params(
          presence = AffineLayer.Params.init(embeddingExtent, Axis[Presence] -> 2, presenceKey),
          construction = AffineLayer.Params.init(embeddingExtent, Axis[IsConstruction] -> 2, constructionKey),
          startX = AffineLayer.Params.init(embeddingExtent, pixelExtent, startXKey),
          startY = AffineLayer.Params.init(embeddingExtent, pixelExtent, startYKey),
          endX = AffineLayer.Params.init(embeddingExtent, pixelExtent, endXKey),
          endY = AffineLayer.Params.init(embeddingExtent, pixelExtent, endYKey),
          midX = AffineLayer.Params.init(embeddingExtent, pixelExtent, midXKey),
          midY = AffineLayer.Params.init(embeddingExtent, pixelExtent, midYKey)
        )
      )
