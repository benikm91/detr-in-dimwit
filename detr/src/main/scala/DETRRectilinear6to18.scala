import dataset.Corpus

/** The detector on the rectilinear parts of six to eighteen lines.
  *
  * `sbt "detr/runMain detrRectilinear6to18Train"` trains it, `detrRectilinear6to18Eval` scores the
  * run it finds, and `detrRectilinear6to18Plot` shows what that run detects.
  *
  * The model is the size the l-shape runs used, deliberately — the corpus is what changes here.
  * The queries are the exception, and they have to change: a drawing holds up to 22 objects
  * against the L's 12, so 32 queries would leave ten slots to cover ten objects with no room to
  * compete. Sixty-four keeps the headroom the L had, in the same proportion.
  */
val DETRRectilinear6to18 = DETRSetup(
  corpus = Corpus.Rectilinear6to18,
  checkpointRoot = "out/detr/rectilinear-6to18",
  numQueries = 64
)

@main
def detrRectilinear6to18Train(): Unit = detrTrain(DETRRectilinear6to18)

@main
def detrRectilinear6to18Eval(): Unit = detrEval(DETRRectilinear6to18)

@main
def detrRectilinear6to18Plot(): Unit = detrPlot(DETRRectilinear6to18)
