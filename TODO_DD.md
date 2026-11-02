# DimWit and DeepWit: features detr-in-dimwit works around

## 1. DeepWit: cached decoding for cross-attention

`MultiHeadCausalSelfAttention.decode(cache, position, embedding)` decodes one position at a time
against a `KVCache`. There is no counterpart for cross-attention onto a source whose keys and values
are projected once (the document, the nodes). Every decoder that writes one token at a time
rebuilds it from `attendAt` and `projectHeads`, and projects the cache by hand:

```scala
// The cache, projected by hand from the attention's weights.
KVCache(document.dot(Axis[Embedding])(params.documentAttention.keyWeights),
        document.dot(Axis[Embedding])(params.documentAttention.valueWeights))

// One decoding step, against it.
val documentRead = MultiHeadUnfusedFullAttention[Patch, Embedding, RelationshipToken, Embedding, V](
  Axis[Patch], Axis[RelationshipToken], params.documentAttention, AttentionScore.scaledDotProduct)
val patches = Tensor1(Axis[Patch]).fromRange(0 until cache.keys.shape(Axis[Patch]))
val query = documentAttentionPreNorm(afterSelf).prependAxis(Axis[RelationshipToken]).dot(Axis[Embedding])(params.documentAttention.queryWeights)
val read = documentRead.projectHeads(documentRead.attendAt(query, cache.keys, cache.values, position.prependAxis(Axis[RelationshipToken]), patches))
val afterDocument = afterSelf + read.squeeze(Axis[RelationshipToken])
```

Wanted, roughly:

```scala
val cache = documentAttention.read(document)                 // KVCache[Patch, V]
val afterDocument = afterSelf + documentAttention.decode(cache, documentAttentionPreNorm(afterSelf))
```

Also for a `MultiHeadCustomAttention` with a mask (point 2), which needs the target position.

Where: `d2tok/src/main/scala/model/WordDecoder.scala` (`WordDecoderBlock.read`, `.decode`),
`d2tok/src/main/scala/floorplangraph/model/RelationshipDecoder.scala` (`RelationshipDecoderBlock.read`,
`.decode`, twice: document and nodes), `d2g/src/main/scala/model/EdgeDecoder.scala`
(`EdgeDecoderBlock.read`, `.decoded`).

## 2. DeepWit: attention where a target position may attend to nothing

With a custom mask that hides every source position from a target position, the softmax divides
zero by zero and the row turns to NaN. It happens when a record has no nodes at all and a decoder
attends onto its nodes. The workaround lets every position through when none is present:

```scala
val readable = where_!(present.any, present, true)
MultiHeadCustomAttention[Node, Embedding, RelationshipToken, Embedding, V](
  params.nodeAttention,
  (words, _) => words.vmap(Axis[RelationshipToken])(_ => readable),
  AttentionScore.scaledDotProduct
)
```

Wanted: a fully masked row reads nothing — zeros — instead of NaN, in `Attention.attend`.

Where: `d2tok/src/main/scala/floorplangraph/model/RelationshipDecoder.scala`
(`RelationshipDecoderBlock.nodeAttention`), `d2g/src/main/scala/model/EdgeDecoder.scala`
(`EdgeDecoderBlock.nodeAttention`).
