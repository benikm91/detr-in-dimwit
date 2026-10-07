package dataset

import java.nio.file.Files
import java.nio.file.Path

/** What a model wrote down for every drawing of the validation split, beside what the drawing
  * holds, so that a question the scores do not answer can be asked afterwards.
  *
  * `<model>-<corpus>-<size>-<step>.jsonl` in [[Runs.outputDir]], one drawing per line, in the
  * order of the split:
  *
  * {{{
  * {"drawing": 0,
  *  "target":    {"nodes": [{"class": "line", "construction": false, "points": [[12.0, 40.0], [80.0, 40.0]]}, ...], "edges": [...]},
  *  "predicted": {"nodes": [...], "edges": [{"class": "connected", "subject": 0, "obj": 3}, ...]},
  *  "matched":   {"0": [null, ...], "2": [1, ...], "4": [1, ...], "8": [1, ...]}}
  * }}}
  *
  * Points are in pixels and in the order [[NodeClass.pointNames]] gives. The predicted nodes are in
  * the order the model wrote them down. `matched` holds, at every tolerance, the target node each
  * predicted node stands for, or `null` — the matching the scores are computed from.
  */
object Transcripts:

  def write(model: String, corpus: Corpus, size: String, split: DrawingDataset.Split, step: Int, drawings: Seq[(RecordGraph, RecordGraph)]): Path =
    def node(at: RecordNode) =
      val points = at.points.map(point => f"[${point.x * corpus.canvas}%.2f, ${point.y * corpus.canvas}%.2f]").mkString("[", ", ", "]")
      s"""{"class": "${at.nodeClass.toString.toLowerCase}", "construction": ${at.isConstruction}, "points": $points}"""
    def edge(at: RecordEdge) =
      s"""{"class": "${at.edgeClass.toString.toLowerCase}", "subject": ${at.subject}, "obj": ${at.obj}}"""
    def record(at: RecordGraph) =
      s"""{"nodes": ${at.nodes.map(node).mkString("[", ", ", "]")}, "edges": ${at.edges.map(edge).mkString("[", ", ", "]")}}"""

    val lines = drawings.zipWithIndex.map:
      case ((target, predicted), drawing) =>
        val matched = Tolerances.map: tolerance =>
          val standsFor = RecordScoring.matching(target.nodes, predicted.nodes, tolerance / corpus.canvas)
          val perNode = predicted.nodes.indices.map(standsFor.get(_).fold("null")(_.toString)).mkString("[", ", ", "]")
          s""""${tolerance.toInt}": $perNode"""
        s"""{"drawing": $drawing, "target": ${record(target)}, "predicted": ${record(predicted)}, "matched": ${matched.mkString("{", ", ", "}")}}"""

    val path = Path.of(Runs.outputDir, s"$model-${corpus.name}-$size-${split.fileName}-$step.jsonl")
    Files.createDirectories(path.getParent)
    Files.writeString(path, lines.mkString("", "\n", "\n"))
