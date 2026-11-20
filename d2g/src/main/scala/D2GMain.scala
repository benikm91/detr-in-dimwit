import d2s.config.D2SModelConfiguration
import dataset.Corpus
import dataset.DrawingDataset.Split
import syntheticFloorplansDataset.FloorplanCorpus

/** Entry points for the D2G experiments, on a corpus (e.g., `rectilinear` or `synthetic-floorplans`)
  * at a model configuration (e.g., `s`): `sbt "d2g/runMain d2gTrain synthetic-floorplans s free free 0"`.
  *
  * Each kind of record has a model of its own: the drawing corpora (see [[Corpus]]) are read by
  * [[d2g.sketch]], the synthetic floor plans (see [[FloorplanCorpus]]) by [[d2g.floorplan]]. The
  * corpus names which, and the orders are that model's: the nodes' one of [[d2s.sketch.NodeOrder.named]]
  * or [[d2s.floorplan.NodeOrder.named]], the relationships' one of [[d2g.sketch.RelationshipOrder.named]]
  * or [[d2g.floorplan.RelationshipOrder.named]]. See [[D2SModelConfiguration]] for the sizes.
  *
  * [[d2gTrain]] trains a run in an order of the nodes and of the relationships, from a seed, so that a
  * setting can be run again.
  * [[d2gTrainFromScratch]] trains every part of a model, nodes and relationships together, from
  * nothing.
  * [[d2gFineTune]] trains every part of a floor plan run further, from one of its checkpoints.
  * [[d2gEval]] scores the newest run of that corpus and size (greedy decoding).
  * [[d2gTest]] scores its last checkpoint on the test split.
  * [[d2gPlot]] shows what a floor plan run transcribes (visualization).
  */

@main
def d2gTrain(corpus: String, size: String, order: String, relationshipOrder: String, seed: Int): Unit =
  floorplans(corpus) match
    case Some(plans) =>
      d2g.floorplan.train.trainTranscriber(floorplanSetup(plans, size).copy(
        nodeOrder = d2s.floorplan.NodeOrder.named(order),
        relationshipOrder = d2g.floorplan.RelationshipOrder.named(relationshipOrder),
        seed = seed
      ))
    case None =>
      d2g.sketch.train.trainTranscriber(sketchSetup(corpus, size).copy(
        nodeOrder = d2s.sketch.NodeOrder.named(order),
        relationshipOrder = d2g.sketch.RelationshipOrder.named(relationshipOrder),
        seed = seed
      ))

@main
def d2gTrainFromScratch(corpus: String, size: String, order: String, relationshipOrder: String, seed: Int): Unit =
  floorplans(corpus) match
    case Some(plans) =>
      d2g.floorplan.train.trainWhole(floorplanSetup(plans, size).copy(
        nodeOrder = d2s.floorplan.NodeOrder.named(order),
        relationshipOrder = d2g.floorplan.RelationshipOrder.named(relationshipOrder),
        seed = seed
      ), startingFrom = None)
    case None =>
      d2g.sketch.train.trainFromScratch(sketchSetup(corpus, size).copy(
        nodeOrder = d2s.sketch.NodeOrder.named(order),
        relationshipOrder = d2g.sketch.RelationshipOrder.named(relationshipOrder),
        seed = seed
      ))

/** As long as training from scratch, at a learning rate that leaves what the model already knows
  * largely in place.
  */
@main
def d2gFineTune(corpus: String, size: String, order: String, relationshipOrder: String, seed: Int, checkpoint: String): Unit =
  d2g.floorplan.train.trainWhole(
    floorplanSetup(FloorplanCorpus.named(corpus), size).copy(
      nodeOrder = d2s.floorplan.NodeOrder.named(order),
      relationshipOrder = d2g.floorplan.RelationshipOrder.named(relationshipOrder),
      seed = seed,
      learningRate = 1e-4f
    ),
    startingFrom = Some(checkpoint)
  )

@main
def d2gEval(corpus: String, size: String): Unit = score(corpus, size, Split.Validation)

@main
def d2gTest(corpus: String, size: String): Unit = score(corpus, size, Split.Test)

@main
def d2gPlot(corpus: String, size: String): Unit = d2g.floorplan.eval.plotTranscriber(floorplanSetup(FloorplanCorpus.named(corpus), size))

private def score(corpus: String, size: String, split: Split): Unit = floorplans(corpus) match
  case Some(plans) => d2g.floorplan.eval.scoreTranscriber(floorplanSetup(plans, size), size, split)
  case None        => d2g.sketch.eval.scoreTranscriber(sketchSetup(corpus, size), size, split)

private def floorplans(corpus: String): Option[FloorplanCorpus] = FloorplanCorpus.values.find(_.name == corpus)

private def floorplanSetup(corpus: FloorplanCorpus, size: String) = d2g.floorplan.config.D2GSetup(corpus, D2SModelConfiguration.named(size))

private def sketchSetup(corpus: String, size: String) = d2g.sketch.config.D2GSetup(Corpus.named(corpus), D2SModelConfiguration.named(size))
