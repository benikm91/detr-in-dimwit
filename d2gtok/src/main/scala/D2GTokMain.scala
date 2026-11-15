import d2g.floorplan.RelationshipOrder
import d2gtok.config.D2GTokSetup
import d2gtok.eval.*
import d2gtok.train.*
import d2tok.config.D2TokModelConfiguration
import d2tok.floorplan.SequenceOrder
import dataset.DrawingDataset.Split
import syntheticFloorplansDataset.FloorplanCorpus

/** Entry points for the D2GTok experiments — d2g's relationships on the nodes of a d2tok run — on a
  * corpus of synthetic floor plans (see [[FloorplanCorpus]]) at a d2tok model configuration (see
  * [[D2TokModelConfiguration]]):
  * `sbt "d2gtok/runMain d2gtokTrain synthetic-floorplans-furnished s-deep walls-first by-kind 0"`.
  * The order of the nodes is the one the d2tok run wrote them in ([[SequenceOrder.named]]), that of
  * the relationships one of [[RelationshipOrder.named]].
  *
  * [[d2gtokTrain]] trains the relationships on the nodes of the d2tok run, which do not move.
  * [[d2gtokFineTune]] trains every part of such a run further, from one of its checkpoints.
  * [[d2gtokEval]] scores the newest run of that corpus and size (greedy decoding).
  * [[d2gtokTest]] scores its last checkpoint on the test split.
  */

@main
def d2gtokTrain(corpus: String, size: String, order: String, relationshipOrder: String, seed: Int): Unit =
  trainTranscriber(writing(setup(corpus, size), order, relationshipOrder, seed))

/** As long as training the relationships, at a learning rate that leaves what the model already
  * knows largely in place.
  */
@main
def d2gtokFineTune(corpus: String, size: String, order: String, relationshipOrder: String, seed: Int, checkpoint: String): Unit =
  fineTuneTranscriber(writing(setup(corpus, size), order, relationshipOrder, seed).copy(learningRate = 1e-4f), checkpoint)

@main
def d2gtokEval(corpus: String, size: String): Unit = scoreTranscriber(setup(corpus, size), size, Split.Validation)

@main
def d2gtokTest(corpus: String, size: String): Unit = scoreTranscriber(setup(corpus, size), size, Split.Test)

private def setup(corpus: String, size: String): D2GTokSetup = D2GTokSetup(FloorplanCorpus.named(corpus), D2TokModelConfiguration.named(size))

/** `setup` writing the nodes in `order` and the relationships in `relationshipOrder`, from `seed`. */
private def writing(setup: D2GTokSetup, order: String, relationshipOrder: String, seed: Int): D2GTokSetup =
  setup.copy(
    nodeWriter = setup.nodeWriter.copy(order = SequenceOrder.named(order)),
    relationshipOrder = RelationshipOrder.named(relationshipOrder),
    seed = seed
  )
