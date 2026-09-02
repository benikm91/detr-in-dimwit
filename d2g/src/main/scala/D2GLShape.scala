import dataset.Corpus

/** The transcription model on the l-shape drawings.
  *
  * `sbt "d2g/runMain d2gLShapeTrain"` trains it, `d2gLShapeEval` scores the run it finds, and
  * `d2gLShapePlot` shows what that run transcribes.
  *
  * The settings here are the ones the corpus converged under: at 500k iterations this reaches
  * 99.0% of nodes, 98.4% of relationships and 94.8% of whole drawings.
  */
val D2GLShape = D2GSetup(
  corpus = Corpus.LShape,
  checkpointRoot = "out/d2g/l-shape"
)

@main
def d2gLShapeTrain(): Unit = d2gTrain(D2GLShape)

@main
def d2gLShapeEval(): Unit = d2gEval(D2GLShape)

@main
def d2gLShapePlot(): Unit = d2gPlot(D2GLShape)
