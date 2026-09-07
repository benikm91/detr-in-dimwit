import dataset.Corpus

/** The transcription model on the rectilinear parts of six to eighteen lines.
  *
  * `sbt "d2g/runMain d2gRectilinear6to18Train"` trains it, `d2gRectilinear6to18Eval` scores the run
  * it finds, and `d2gRectilinear6to18Plot` shows what that run transcribes.
  *
  * The model is the size the l-shape runs used, deliberately: the corpus is what changes here, and
  * a model that changed with it would leave the two unable to be compared. Only the record slots
  * grow, and they grow because the corpus says so — 22 nodes and 22 relationships against the L's
  * 12, so the sequence read and written is nearly twice as long.
  */
val D2GRectilinear6to18 = D2GSetup(
  corpus = Corpus.Rectilinear6to18,
  checkpointRoot = "out/d2g/rectilinear-6to18",
  numLayers = 3,
  numHeads = 8,
  embedding = 256,
  numIterations = 200_000
)

@main
def d2gRectilinear6to18Train(): Unit = d2gTrain(D2GRectilinear6to18)

@main
def d2gRectilinear6to18Eval(): Unit = d2gEval(D2GRectilinear6to18)

@main
def d2gRectilinear6to18Plot(): Unit = d2gPlot(D2GRectilinear6to18)

/** Every query of the pool on its own. Each is trained to answer with a node the others did not, so
  * none of them is the model's primary transcriber and there is no reason the first should be the
  * best one to follow.
  */
@main
def d2gRectilinear6to18EvalEachQuery(): Unit =
  (0 until D2GRectilinear6to18.queryPool).foreach(query => d2gEval(D2GRectilinear6to18, query))
