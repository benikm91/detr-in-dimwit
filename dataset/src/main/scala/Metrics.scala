package dataset

import java.nio.file.Files
import java.nio.file.Path

/** What a run is measured by: every checkpoint, at every tolerance, scored on the validation
  * split — and at every threshold, where the model decides by one.
  */
object Metrics:

  case class Row(step: Int, threshold: Option[Float], tolerance: Float, scored: Seq[RecordScoring.Scored])

  private val Header =
    "model,corpus,size,step,threshold,tolerance,parameters,training_seconds," +
      "node_recall,node_precision,nodes_exact,edge_recall,edge_precision,records_exact"

  /** Writes `<model>-<corpus>-<size>.csv` into [[Runs.outputDir]] and says where. A percentage
    * over nothing — a detector's relationships — is left blank.
    */
  def write(model: String, corpus: Corpus, size: String, parameters: Int, trainingSeconds: Option[Long], rows: Seq[Row]): Path =
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
    val path = Path.of(Runs.outputDir, s"$model-${corpus.name}-$size.csv")
    Files.createDirectories(path.getParent)
    Files.writeString(path, (Header +: lines).mkString("", "\n", "\n"))
    path
