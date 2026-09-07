import dataset.Corpus

/** The transcription model on the l-shape drawings.
  *
  * `sbt "d2g/runMain d2gLShapeTrain"` trains it, `d2gLShapeEval` scores the run it finds, and
  * `d2gLShapePlot` shows what that run transcribes.
  */
val D2GLShape = D2GSetup(
  corpus = Corpus.LShape,
  checkpointRoot = "out/d2g/l-shape"
)

@main
def d2gLShapeTrain(): Unit = d2gTrain(D2GLShape)

@main
def d2gLShapeEval(): Unit = d2gEval(D2GLShape)

/** The same, transcribed by the second query — which answers with the node the first one did not,
  * so the record is written down in a different order.
  */
@main
def d2gLShapeEvalSecond(): Unit = d2gEval(D2GLShape, query = 1)

@main
def d2gLShapePlot(): Unit = d2gPlot(D2GLShape)

/** Every query of the pool on its own, so that following one is measured for all of them rather
  * than for whichever happens to be first.
  */
@main
def d2gLShapeEvalEachQuery(): Unit = (0 until D2GLShape.queryPool).foreach(query => d2gEval(D2GLShape, query))
