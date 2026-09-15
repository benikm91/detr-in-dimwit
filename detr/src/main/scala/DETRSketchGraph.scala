import dataset.Corpus

/** The detector on the CAD sketches.
  *
  * `sbt "detr/runMain detrSketchGraphTrain"` trains it, `detrSketchGraphEval` scores the run it
  * finds, and `detrSketchGraphPlot` shows what that run detects.
  *
  * The drawings are the first this codebase reads that were not generated for it, and the first
  * that hold a circle: what a detector answers with here is a box around a line or around a
  * circle, and nothing is annotated, so the text class never comes up.
  *
  * Sixteen entities against the rectilinear parts' twenty-two, and the queries keep the headroom
  * the other corpora have — a little under three per object a drawing can hold.
  */
val DETRSketchGraph = DETRSetup(
  corpus = Corpus.SketchGraph,
  checkpointRoot = "out/detr/sketch-graph",
  numLayers = 3,
  numHeads = 8,
  embedding = 256,
  numIterations = 200_000,
  numQueries = 48
)

@main
def detrSketchGraphTrain(): Unit = detrTrain(DETRSketchGraph)

@main
def detrSketchGraphEval(): Unit = detrEval(DETRSketchGraph)

@main
def detrSketchGraphPlot(): Unit = detrPlot(DETRSketchGraph)
