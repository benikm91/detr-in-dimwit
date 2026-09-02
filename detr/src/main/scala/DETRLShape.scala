import dataset.Corpus

/** The detector on the l-shape drawings.
  *
  * `sbt "detr/runMain detrLShapeTrain"` trains it, `detrLShapeEval` scores the run it finds, and
  * `detrLShapePlot` shows what that run detects.
  */
val DETRLShape = DETRSetup(
  corpus = Corpus.LShape,
  checkpointRoot = "out/detr/l-shape",
  numQueries = 32
)

@main
def detrLShapeTrain(): Unit = detrTrain(DETRLShape)

@main
def detrLShapeEval(): Unit = detrEval(DETRLShape)

@main
def detrLShapePlot(): Unit = detrPlot(DETRLShape)
