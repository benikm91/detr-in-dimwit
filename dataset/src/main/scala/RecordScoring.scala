package dataset

/** How far a coordinate may be off, in pixels. Zero is exact, which a record read off a discrete
  * prediction can be held to.
  */
val Tolerances = Seq(0f, 2f, 4f, 8f)

/** Scoring a predicted record against the record a drawing was rendered from.
  *
  * A record is a set, so the two are not compared slot by slot: nodes are matched by what they
  * say, and relationships are then compared by the nodes they name rather than by the slots they
  * name them with. A record written down in a different order is the same record.
  */
object RecordScoring:

  /** One drawing, scored. `isExact` is the whole record at once: every node, every relationship,
    * nothing missing and nothing spurious.
    */
  case class Scored(
      nodes: Int,
      nodesFound: Int,
      nodesPredicted: Int,
      relationships: Int,
      relationshipsFound: Int,
      relationshipsPredicted: Int,
      isExact: Boolean
  )

  def score(target: RecordGraph, found: RecordGraph, tolerance: Float): Scored =
    val matched = matching(target.nodes, found.nodes, tolerance)
    val wanted = multiset(target.edges.map(canonical))
    val present = multiset(found.edges.flatMap(resolve(matched)).map(canonical))
    Scored(
      nodes = target.nodes.length,
      nodesFound = matched.size,
      nodesPredicted = found.nodes.length,
      relationships = target.edges.length,
      relationshipsFound = wanted.map((edge, count) => math.min(count, present.getOrElse(edge, 0))).sum,
      relationshipsPredicted = found.edges.length,
      isExact = matched.size == target.nodes.length &&
        found.nodes.length == target.nodes.length &&
        found.edges.length == present.values.sum &&
        present == wanted
    )

  /** Which found node stands for which target node. Greedy, which is exact here: the nodes of a
    * drawing are further apart than any tolerance scored, so a target has at most one in reach.
    */
  private def matching(target: Seq[RecordNode], found: Seq[RecordNode], tolerance: Float): Map[Int, Int] =
    target.zipWithIndex.foldLeft(Map.empty[Int, Int]):
      case (matched, (node, index)) =>
        found.indices
          .find(slot => !matched.contains(slot) && isFound(node, found(slot), tolerance))
          .fold(matched)(slot => matched + (slot -> index))

  private def isFound(target: RecordNode, found: RecordNode, tolerance: Float): Boolean =
    target.nodeClass == found.nodeClass &&
      target.points.zip(found.points).forall((wanted, at) => (wanted.x - at.x).abs <= tolerance && (wanted.y - at.y).abs <= tolerance)

  /** A found relationship over the target's nodes, or nothing when it names a node that was not
    * found: an edge between nodes that are not there relates nothing.
    */
  private def resolve(matched: Map[Int, Int])(edge: RecordEdge): Option[RecordEdge] =
    for
      subject <- matched.get(edge.subject)
      obj <- matched.get(edge.obj)
    yield RecordEdge(edge.edgeClass, subject, obj)

  /** A symmetric relationship says nothing by which of its two nodes comes first. */
  private def canonical(edge: RecordEdge): RecordEdge =
    if edge.edgeClass.isSymmetric && edge.subject > edge.obj then RecordEdge(edge.edgeClass, edge.obj, edge.subject) else edge

  private def multiset(edges: Seq[RecordEdge]): Map[RecordEdge, Int] =
    edges.groupBy(identity).view.mapValues(_.length).toMap

/** One line of a score report: `<prefix>  <what>  <correct> / <total>  <percentage>`. */
def report(prefix: String, what: String, correct: Int, total: Int): Unit =
  println(f"$prefix%5s  $what%-26s $correct%6d / $total%-6d ${100f * correct / total}%5.1f%%")

/** `tolerance px`, the [[report]] prefix of a score that depends on the tolerance. */
def at(tolerance: Float): String = f"$tolerance%2.0f px"
