package dataset

import dataset.LShapeDataset.Split
import dimwit.*
import plotwit.*
import viz.PlotTargets.websocket

private trait Width derives Label
private trait Height derives Label
private trait Channel derives Label
private trait Node derives Label

/** Shows what [[LShapeDataset]] hands out: `sbt "dataset/runMain dataset.lShapeDemo"`.
  *
  * Reads the validation split (67 MB, downloaded into the HuggingFace cache on first run) and
  * plots the first few drawings next to the same drawings with the boxes their records are drawn
  * as. The records themselves have no drawing, so they are printed.
  */
@main
def lShapeDemo(): Unit =
  dimwit.initialize()

  val data = LShapeDataset.open(Axis[Width], Axis[Height], Axis[Channel], Axis[Node])(Split.Validation)
  println(data)
  println(s"most nodes a record holds: ${data.observedMaxNodes}")
  println(s"links naming something the record does not hold: ${data.unresolvedLinks}")

  val rows = data.samples().take(3).zipWithIndex.map: (sample, index) =>
    val drawing = Outlines.greyLevels(sample.image)
    val objects = Objects.of(sample.target, data.geometry)
    println(s"sample $index: ${describe(RecordGraph.of(sample.target))}")
    Seq(
      plots.imagePlot(drawing, _.title := s"sample $index"),
      plots.imagePlot(Outlines(drawing, objects.detection), _.title := s"sample $index — objects")
    )

  display(grid(rows.toSeq))

private def describe(record: RecordGraph): String =
  val nodes = record.nodes.map(node => s"${node.nodeClass}(${node.points.map(point => f"${point.x}%.3f, ${point.y}%.3f").mkString("; ")})")
  val edges = record.edges.map(edge => s"${edge.edgeClass}(${edge.subject}, ${edge.obj})")
  (nodes ++ edges).mkString(", ")
