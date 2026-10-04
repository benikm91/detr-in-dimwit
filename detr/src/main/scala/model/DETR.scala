package detr.model

import detr.*
import detr.train.*
import detr.eval.*
import detr.config.*
import dataset.NodeClass
import dataset.NodeClasses
import dataset.RecordNodes
import deepwit.base.AffineLayer
import documentEncoder.DocumentEncoder
import deepwit.normalization.LayerNorm
import deepwit.init.Init
import dimwit.*
import dimwit.tensor.Tensor4

/** DETR, [[https://arxiv.org/abs/2005.12872 End-to-End Object Detection with Transformers]],
  * with a vision transformer in place of the convolutional backbone — see `README.md` for
  * the divergences from the paper.
  *
  * The image is embedded patch by patch and attended over by the encoder. The decoder turns
  * a fixed set of learned object queries into one embedding per [[Query]], from which the
  * [[NodeHead]] predicts a node of the record, or none.
  */
class DETR[V: IsFloating](params: DETR.Params[V]) extends (Tensor3[Width, Height, Channel, V] => RecordNodes[Query]):

  private val encodeDocument = DocumentEncoder(params.encoder)
  private val decoder = DETRDecoder(Axis[Patch], Axis[Query], params.decoder)
  private val nodeHead = NodeHead(params.nodeHead)

  /** The node every query answers with, [[NodeClass.NoNode]] where it answers with none. */
  override def apply(image: Tensor3[Width, Height, Channel, V]): RecordNodes[Query] =
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
      val (classKey, startXKey, startYKey, endXKey, endYKey, midXKey, midYKey) = headsKey.splitToTuple(7)
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
          nodeClass = AffineLayer.Params.init(embeddingExtent, Axis[NodeClasses] -> NodeClass.values.length, classKey),
          startX = AffineLayer.Params.init(embeddingExtent, pixelExtent, startXKey),
          startY = AffineLayer.Params.init(embeddingExtent, pixelExtent, startYKey),
          endX = AffineLayer.Params.init(embeddingExtent, pixelExtent, endXKey),
          endY = AffineLayer.Params.init(embeddingExtent, pixelExtent, endYKey),
          midX = AffineLayer.Params.init(embeddingExtent, pixelExtent, midXKey),
          midY = AffineLayer.Params.init(embeddingExtent, pixelExtent, midYKey)
        )
      )

  /** How big a model a run asks for, which is the one thing about the model a run gets to choose.
    * Everything else is fixed, so that two runs of the same size are the same model.
    */
  /** `s-deep` is the transformer of [[https://arxiv.org/abs/2005.12872 DETR]] itself: six layers
    * either side, 256 wide, eight heads.
    */
  enum Size(val name: String, val embedding: Int, val numLayers: Int, val numHeads: Int):
    case XS extends Size("xs", 128, 3, 4)
    case S extends Size("s", 256, 3, 8)
    case SDeep extends Size("s-deep", 256, 6, 8)

  object Size:

    /** The size a run names on its command line. */
    def named(name: String): Size =
      values.find(_.name == name).getOrElse(sys.error(s"no size named '$name': ${values.map(_.name).mkString(", ")}"))
