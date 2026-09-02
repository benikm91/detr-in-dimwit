import dataset.Corpus

/** The scene graph model on the l-shape drawings.
  *
  * `sbt "egtr/runMain egtrLShapeTrain"` trains it — pass a run directory to start from a
  * particular detector — `egtrLShapeEval` scores the run it finds.
  */
val EGTRLShape = EGTRSetup(
  corpus = Corpus.LShape,
  checkpointRoot = "out/egtr/l-shape",
  numQueries = 32,
  detectorCheckpointRoot = Some(s"../detr/${DETRLShape.checkpointRoot}")
)

@main
def egtrLShapeTrain(detectorRun: String*): Unit = egtrTrain(EGTRLShape, detectorRun.headOption)

@main
def egtrLShapeEval(): Unit = egtrEval(EGTRLShape)
