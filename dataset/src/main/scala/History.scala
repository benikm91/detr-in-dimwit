package dataset

import java.nio.file.Files
import java.nio.file.Path
import java.nio.file.StandardOpenOption.APPEND
import java.time.LocalDateTime
import java.time.format.DateTimeFormatter

/** How a run went while it was running: a line per checkpoint, written as the checkpoint is, so
  * that a run can be watched and a run cut short still says how far it got.
  *
  * The loss is the running average the run reports as it goes, not a mean over the whole of it.
  */
final class History private (path: Path):

  def add(step: Int, trainLoss: Float): Unit =
    Files.writeString(path, f"$step,${LocalDateTime.now.format(History.When)},$trainLoss%.4f\n", APPEND)

object History:

  private val Header = "step,time,train_loss"

  private val When = DateTimeFormatter.ofPattern("yyyy-MM-dd HH:mm:ss")

  /** `history.csv` in [[Runs.outputDir]], emptied of whatever was there. */
  def apply(): History =
    val path = Path.of(Runs.outputDir, "history.csv")
    Files.createDirectories(path.getParent)
    Files.writeString(path, s"$Header\n")
    new History(path)
