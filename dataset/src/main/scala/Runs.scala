package dataset

import dimwit.*
import dimwit.tensortree.TensorTree

import java.nio.file.Files
import java.nio.file.Path

/** Where a run keeps what it makes: checkpoints under `CHECKPOINT_DIR` and what is measured of
  * them under `OUTPUT_DIR`. The cluster sets both; a desk leaves both at `out`.
  */
object Runs:

  val checkpointDir: String = sys.env.getOrElse("CHECKPOINT_DIR", "out")
  val outputDir: String = sys.env.getOrElse("OUTPUT_DIR", "out")

  /** How long a run trained for, kept beside its checkpoints so that scoring them can say. A run
    * that takes more than one job adds what each of them spent to what is already noted.
    */
  def noteTrainingSeconds(runDir: String, seconds: Long): Unit =
    val sofar = trainingSeconds(runDir).getOrElse(0L)
    Files.writeString(Path.of(runDir, "training-seconds"), (sofar + seconds).toString)

  def trainingSeconds(runDir: String): Option[Long] =
    val noted = Path.of(runDir, "training-seconds")
    Option.when(Files.isRegularFile(noted))(Files.readString(noted).trim.toLong)

  private trait Parameter derives Label

  /** How many numbers a model is made of. */
  def parameters[Params: TensorTree](params: Params): Int =
    val (flatten, _) = TensorTree.ravel(params, Axis[Parameter])
    flatten(params).shape(Axis[Parameter])
