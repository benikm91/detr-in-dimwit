package egtr

import dataset.RelationClasses
import deepwit.activation.relu
import deepwit.activation.sigmoid
import deepwit.attention.Head
import deepwit.attention.HeadKey
import deepwit.attention.HeadQuery
import deepwit.base.AffineLayer
import deepwit.base.LinearLayer
import detr.*
import dimwit.*
import dimwit.Conversions.given
import dimwit.Label as Λ
import deepwit.init.Init

import scala.language.implicitConversions

/** The relation extractor of [[https://arxiv.org/abs/2404.02072 EGTR]], equations 3 to 6.
  *
  * The decoder's self-attention already relates the object queries to each other: to decide
  * what it stands for, every query reads every other one, and the queries and keys it does so
  * by are exactly a statement about the pair. EGTR's point is that this by-product is most of a
  * scene graph already, so the extractor is a shallow head on top of it rather than a second
  * decoder with relation queries of its own.
  *
  * Each of the decoder's blocks contributes one source, and the final object embeddings one
  * more. A source gives every query a subject and an object half — the attention query in the
  * role of the subject, the attention key in the role of the object (eq. 3), and two learned
  * projections of the embedding for the last source (eq. 4). Concatenated over an ordered pair
  * of queries, that is one relation representation per pair per source.
  *
  * The sources are then summed rather than concatenated, each weighted by a gate it computes
  * for itself (eq. 5), so that a pair can be decided from whichever block relates it most
  * clearly without the head growing with the depth of the decoder. Two three-layer perceptrons
  * read the sum: one scores every [[dataset.RelationClass]] of the pair, the other scores
  * whether the pair is connected at all (§3.3.2), which multiplies into the score at inference
  * to suppress pairs no relation was meant for.
  */
class RelationExtractor[V: IsFloating](params: RelationExtractor.Params[V])
    extends (DETR.Decoded[V] => RelationExtractor.Graph[V]):

  import RelationExtractor.Graph
  import RelationExtractor.Pairs
  import RelationExtractor.Perceptron

  override def apply(decoded: DETR.Decoded[V]): Graph[V] =
    // Eq. 6: the gated sum over all sources, which both heads read.
    val gated = sources(decoded).map(pairs => pairs *! gate(pairs)).reduce(_ + _)
    Graph(
      relationLogits = perceptron(params.relation)(gated),
      connectivityLogits = perceptron(params.connectivity)(gated).slice(Axis[Connectivity].at(0))
    )

  /** One relation representation per ordered pair of queries, per source: the self-attention of
    * every decoder block (eq. 3) and the object embeddings the detection heads read (eq. 4).
    */
  private def sources(decoded: DETR.Decoded[V]): List[Pairs[V]] =
    require(
      params.attention.size == decoded.selfAttention.size,
      s"the extractor reads ${params.attention.size} decoder blocks but the detector has ${decoded.selfAttention.size}"
    )
    val attention = params.attention.zip(decoded.selfAttention).map: (projection, source) =>
      pairs(project(projection.subject, overHeads(source.queries)), project(projection.obj, overHeads(source.keys)))
    val embedding = params.embedding
    attention :+ pairs(project(embedding.subject, decoded.objects), project(embedding.obj, decoded.objects))

  /** The heads of one self-attention concatenated into the single space eq. 3 reads a pair from.
    *
    * Every head relates the object queries on a query and key space of its own, so what stands
    * for a pair is all of them at once — the whole `d` wide space the block splits into heads.
    */
  private def overHeads[HeadSpace: Λ](perHead: Tensor3[BoundingBox, Head, HeadSpace, V]): Tensor2[BoundingBox, Head |*| HeadSpace, V] =
    perHead.vmap(Axis[BoundingBox])(_.flatten)

  /** The subject and object halves of one source, concatenated over every ordered pair. */
  private def pairs(subject: Tensor2[BoundingBox, RelationSource, V], obj: Tensor2[BoundingBox, RelationSource, V]): Pairs[V] =
    val half = Shape3(
      subject.shape.extent(Axis[BoundingBox]),
      Axis[RelatedBox] -> subject.shape(Axis[BoundingBox]),
      subject.shape.extent(Axis[RelationSource])
    )
    concatenate(
      subject.broadcastTo(half),
      obj.vmap(Axis[RelationSource])(_.relabelTo(Axis[RelatedBox])).broadcastTo(half),
      Axis[RelationSource]
    )

  private def project[In: Λ](params: LinearLayer.Params[In, RelationSource, V], source: Tensor2[BoundingBox, In, V]): Tensor2[BoundingBox, RelationSource, V] =
    source.dot(Axis[In])(params.weight)

  /** Eq. 5: how much of a source's representation of a pair flows into the sum. */
  private def gate(pairs: Pairs[V]): Tensor2[BoundingBox, RelatedBox, V] =
    sigmoid(pairs.dot(Axis[RelationSource])(params.gate))

  private def perceptron[Out: Λ](params: Perceptron[Out, V])(pairs: Pairs[V]): Tensor3[BoundingBox, RelatedBox, Out, V] =
    affine(params.output)(relu(affine(params.hidden2)(relu(affine(params.hidden1)(pairs)))))

  /** An affine layer over the last axis of the grid, i.e. applied to every pair alike. */
  private def affine[In: Λ, Out: Λ](params: AffineLayer.Params[In, Out, V])(pairs: Tensor3[BoundingBox, RelatedBox, In, V]): Tensor3[BoundingBox, RelatedBox, Out, V] =
    pairs.dot(Axis[In])(params.weight) +! params.bias

object RelationExtractor:

  /** One relation representation per ordered pair of query slots. */
  type Pairs[V] = Tensor3[BoundingBox, RelatedBox, RelationSource, V]

  /** What the extractor scores, before deciding.
    *
    * @param relationLogits     Per ordered pair, the score of every [[dataset.RelationClass]].
    *                           Sigmoid rather than softmax: a pair may carry several relations,
    *                           or none.
    * @param connectivityLogits Per ordered pair, the score of it carrying any relation at all.
    */
  case class Graph[V](
      relationLogits: Tensor3[BoundingBox, RelatedBox, RelationClasses, V],
      connectivityLogits: Tensor2[BoundingBox, RelatedBox, V]
  )

  /** Eq. 3: what gives a block's attention queries and keys the roles of subject and object. */
  case class AttentionProjection[V](
      subject: LinearLayer.Params[Head |*| HeadQuery, RelationSource, V],
      obj: LinearLayer.Params[Head |*| HeadKey, RelationSource, V]
  )

  /** Eq. 4: the same for the object embeddings, which carry no such roles yet. */
  case class EmbeddingProjection[V](
      subject: LinearLayer.Params[DETR.Embedding, RelationSource, V],
      obj: LinearLayer.Params[DETR.Embedding, RelationSource, V]
  )

  /** `MLP_rel` and `MLP_con`: a three layer perceptron with ReLU, as the box head of DETR. */
  case class Perceptron[Out, V](
      hidden1: AffineLayer.Params[RelationSource, RelationHidden, V],
      hidden2: AffineLayer.Params[RelationHidden, Prime[RelationHidden], V],
      output: AffineLayer.Params[Prime[RelationHidden], Out, V]
  )

  /** @param attention    One projection per decoder block, in the order the blocks run.
    * @param embedding    The projection of the final object embeddings.
    * @param gate         Eq. 5: the single gate weight, shared by every source.
    * @param relation     `MLP_rel`, scoring the relations of a pair.
    * @param connectivity `MLP_con`, scoring whether a pair is connected at all.
    */
  case class Params[V](
      attention: List[AttentionProjection[V]],
      embedding: EmbeddingProjection[V],
      gate: Tensor1[RelationSource, V],
      relation: Perceptron[RelationClasses, V],
      connectivity: Perceptron[Connectivity, V]
  )

  object Params:

    /** Sizes an extractor to the detector it reads.
      *
      * @param sourceExtent The width of one half of a relation representation, `d_model` in the
      *                     paper. The representation of a pair is twice this.
      * @param hiddenExtent The width of the hidden layers of both heads.
      */
    def xavierUniform[V: IsFloating](
        detector: DETR.Params[V],
        sourceExtent: AxisExtent[RelationSource],
        hiddenExtent: AxisExtent[RelationHidden],
        key: Key,
        vtype: VType[V] = VType[Float32]
    ): Params[V] =
      val (attentionKey, embeddingKey, gateKey, relationKey, connectivityKey) = key.splitToTuple(5)
      val blocks = detector.decoder.decoderBlocks
      val attention = blocks.head.selfAttentionParams.multiHeadAttention
      val queryExtent = Axis[Head |*| HeadQuery] -> attention.queryWeights.shape(Axis[Head]) * attention.queryWeights.shape(Axis[HeadQuery])
      val keyExtent = Axis[Head |*| HeadKey] -> attention.keyWeights.shape(Axis[Head]) * attention.keyWeights.shape(Axis[HeadKey])
      val embeddingExtent = Axis[DETR.Embedding] -> detector.objectQueries.shape(Axis[DETR.Embedding])
      // A representation is the two halves concatenated, which is what both heads and the gate read.
      val pairExtent = Axis[RelationSource] -> sourceExtent.size * 2
      Params(
        attention = attentionKey
          .split(blocks.size)
          .map: key =>
            val (subjectKey, objectKey) = key.splitToTuple(2)
            AttentionProjection(
              subject = LinearLayer.Params.xavierUniform(queryExtent, sourceExtent, subjectKey, vtype),
              obj = LinearLayer.Params.xavierUniform(keyExtent, sourceExtent, objectKey, vtype)
            )
          .toList,
        embedding =
          val (subjectKey, objectKey) = embeddingKey.splitToTuple(2)
          EmbeddingProjection(
            subject = LinearLayer.Params.xavierUniform(embeddingExtent, sourceExtent, subjectKey, vtype),
            obj = LinearLayer.Params.xavierUniform(embeddingExtent, sourceExtent, objectKey, vtype)
          )
        ,
        gate = Init.xavierUniform(pairExtent, Axis[Gate] -> 1, gateKey, vtype).slice(Axis[Gate].at(0)),
        relation = perceptron(pairExtent, hiddenExtent, Axis[RelationClasses] -> dataset.RelationClass.values.length, relationKey, vtype),
        connectivity = perceptron(pairExtent, hiddenExtent, Axis[Connectivity] -> 1, connectivityKey, vtype)
      )

    private def perceptron[Out: Λ, V: IsFloating](
        pairExtent: AxisExtent[RelationSource],
        hiddenExtent: AxisExtent[RelationHidden],
        outExtent: AxisExtent[Out],
        key: Key,
        vtype: VType[V] = VType[Float32]
    ): Perceptron[Out, V] =
      val (hidden1Key, hidden2Key, outputKey) = key.splitToTuple(3)
      Perceptron(
        hidden1 = AffineLayer.Params.xavierUniform(pairExtent, hiddenExtent, hidden1Key, vtype),
        hidden2 = AffineLayer.Params.xavierUniform(hiddenExtent, Axis[Prime[RelationHidden]] -> hiddenExtent.size, hidden2Key, vtype),
        output = AffineLayer.Params.xavierUniform(Axis[Prime[RelationHidden]] -> hiddenExtent.size, outExtent, outputKey, vtype)
      )

    /** Axis the gate weight is initialized along, so that it is scaled as a layer into one
      * score rather than as a bare vector.
      */
    private trait Gate derives Label
