import dataset.Corpus

/** The scene graph model on the rectilinear parts of six to eighteen lines.
  *
  * `sbt "egtr/runMain egtrRectilinear6to18Train"` trains it, `egtrRectilinear6to18Eval` scores the
  * run it finds.
  *
  * Trained from scratch rather than from the detector this corpus already has. The transcription
  * model is given nothing but the drawings, so a scene graph model started from a detector that
  * has already been through the corpus once would not be answering the same question — it would
  * have seen the training split twice over. Pass a run directory to start from one anyway.
  */
val EGTRRectilinear6to18 = EGTRSetup(
  corpus = Corpus.Rectilinear6to18,
  checkpointRoot = "out/egtr/rectilinear-6to18",
  numQueries = 64,
  detectorCheckpointRoot = None
)

@main
def egtrRectilinear6to18Train(detectorRun: String*): Unit =
  egtrTrain(EGTRRectilinear6to18, detectorRun.headOption)

@main
def egtrRectilinear6to18Eval(): Unit = egtrEval(EGTRRectilinear6to18)
