package dataset

import java.nio.file.Files
import java.nio.file.Path
import java.nio.file.StandardOpenOption

/** What a run is measured by: every checkpoint, at every tolerance, scored on the validation
  * split — and at every threshold, where the model decides by one.
  */
object Metrics:

  case class Row(step: Int, threshold: Option[Float], tolerance: Float, scored: Seq[RecordScoring.Scored])

  private val Header =
    "model,corpus,size,step,threshold,tolerance,parameters,training_seconds," +
      "node_recall,node_precision,nodes_exact,edge_recall,edge_precision,records_exact"

  /** `<model>-<corpus>-<size>.csv` in [[Runs.outputDir]], begun with its header alone. Rows are
    * appended as each checkpoint is scored, so a run cut short keeps the checkpoints it reached. A
    * percentage over nothing — a detector's relationships — is left blank.
    */
  class Csv(model: String, corpus: Corpus, size: String, parameters: Int, trainingSeconds: Option[Long]):
    val path = Path.of(Runs.outputDir, s"$model-${corpus.name}-$size.csv")
    Files.createDirectories(path.getParent)
    Files.writeString(path, Header + "\n")

    def append(rows: Seq[Row]): Unit =
      def percent(correct: Int, total: Int) = if total == 0 then "" else f"${100f * correct / total}%.2f"
      val lines = rows.map: row =>
        val scored = row.scored
        Seq(
          model,
          corpus.name,
          size,
          row.step,
          row.threshold.fold("")(_.toString),
          row.tolerance.toInt,
          parameters,
          trainingSeconds.fold("")(_.toString),
          percent(scored.map(_.nodesFound).sum, scored.map(_.nodes).sum),
          percent(scored.map(_.nodesFound).sum, scored.map(_.nodesPredicted).sum),
          percent(scored.count(_.nodesExact), scored.length),
          percent(scored.map(_.relationshipsFound).sum, scored.map(_.relationships).sum),
          percent(scored.map(_.relationshipsFound).sum, scored.map(_.relationshipsPredicted).sum),
          percent(scored.count(_.isExact), scored.length)
        ).mkString(",")
      Files.writeString(path, lines.mkString("", "\n", "\n"), StandardOpenOption.APPEND)
