package dataset

import java.nio.file.Files
import java.nio.file.Path
import java.nio.file.StandardOpenOption.APPEND
import java.nio.file.StandardOpenOption.CREATE
import java.time.LocalDateTime
import java.time.format.DateTimeFormatter

/** How a run went while it was running: a line per checkpoint, written as the checkpoint is, so
  * that a run can be watched and a run cut short still says how far it got. It is kept beside the
  * checkpoints, and a run that takes more than one job carries on the lines of the jobs before it.
  *
  * The loss is the running average the run reports as it goes, not a mean over the whole of it.
  */
final class History(runDir: String):

  private val path = Path.of(runDir, "history.csv")

  /** Call it once the checkpoint is saved: the run's folder only exists from its first one. */
  def add(step: Int, trainLoss: Float): Unit =
    val header = if Files.isRegularFile(path) then "" else s"${History.Header}\n"
    Files.writeString(path, f"$header$step,${LocalDateTime.now.format(History.When)},$trainLoss%.4f\n", CREATE, APPEND)

object History:

  private val Header = "step,time,train_loss"

  private val When = DateTimeFormatter.ofPattern("yyyy-MM-dd HH:mm:ss")
