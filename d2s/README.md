# D2S — document-to-set transcription

The node half of [d2g](../d2g): a drawing is transcribed into the **set of nodes** it holds, one
node at a time, each answered from the nodes not yet taken. Nothing is predicted about how the
nodes relate. That makes it the model to compare with [detr](../detr), which predicts the same
set in one pass, and d2g is a D2S with the relationships added on top.

```
sbt "d2s/runMain d2sTrain sketch-xl s-deep 0" # trains a run from seed 0
sbt "d2s/runMain d2sEval sketch-xl s-deep"    # scores every checkpoint of the newest run
sbt "d2s/runMain d2sDraw sketch-xl s-deep"    # draws what every checkpoint transcribed
```

On the cluster, `runs/queue_d2s.sh <corpus> <size>`.

A record's relationships are read with it and left aside: a set is scored against the nodes of
the record it was rendered from, as the detector is.

## Files

| | |
|---|---|
| [D2S.scala](src/main/scala/model/D2S.scala) | the model: the encoder, then the nodes |
| [NodeDecoder.scala](src/main/scala/model/NodeDecoder.scala) | the decoder of the nodes, and the mask its sequence reads by |
| [NodeEmbedder.scala](src/main/scala/model/NodeEmbedder.scala), [NodeHead.scala](src/main/scala/model/NodeHead.scala) | a node into an embedding, and back into a node |
| [RemainingNodeLoss.scala](src/main/scala/train/RemainingNodeLoss.scala) | the remaining-node loss, and the costs the relationships' loss in d2g shares |
| [D2STrain.scala](src/main/scala/train/D2STrain.scala) | the training loop |
| [D2SEval.scala](src/main/scala/eval/D2SEval.scala) | the transcriber, scoring and plotting |
| [D2SSetup.scala](src/main/scala/config/D2SSetup.scala) | a run's setup, and the model sizes d2s and d2g share |
