import deepwit.checkpointing.TensorTreeCheckpointer

/** The checkpoints of the run named on the command line, or of the newest run under `root`. */
def checkpointsIn(root: String, run: Seq[String]): TensorTreeCheckpointer =
  run.headOption
    .map(TensorTreeCheckpointer(_))
    .orElse(TensorTreeCheckpointer.latestIn(root))
    .getOrElse(sys.error(s"no training run in $root"))
