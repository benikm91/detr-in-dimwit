import detr.*
import detr.model.*
import detr.train.*
import detr.eval.*
import detr.config.*
import egtr.*
import egtr.model.*
import egtr.train.*
import egtr.eval.*
import egtr.config.*
import dataset.Corpus

/** The scene graph model, on a corpus at a size — `sbt "egtr/runMain egtrTrain sketch s"`.
  *
  * [[egtrTrain]] trains a run from scratch and [[egtrEval]] scores the newest run of that corpus
  * and size. See [[Corpus]] and [[EGTR.Size]] for the names.
  */
@main
def egtrTrain(corpus: String, size: String): Unit = trainSceneGraph(EGTRSetup(Corpus.named(corpus), EGTR.Size.named(size)))

@main
def egtrEval(corpus: String, size: String): Unit = scoreSceneGraph(EGTRSetup(Corpus.named(corpus), EGTR.Size.named(size)), size)
