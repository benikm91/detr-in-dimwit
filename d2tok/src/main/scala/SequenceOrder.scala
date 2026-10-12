package d2tok

/** The order a record's nodes are written in. A sequence model learns to write one node after the
  * other, so the order of its targets is the order it learns.
  */
enum SequenceOrder(val name: String):

  /** A fresh random order every time a record is seen, as for a collection with no order of its own. */
  case Random extends SequenceOrder("random")

  /** The order the corpus stores a record's nodes in. */
  case Kept extends SequenceOrder("kept")

object SequenceOrder:
  def named(name: String): SequenceOrder = values.find(_.name == name).getOrElse(sys.error(s"no order named '$name': ${values.map(_.name).mkString(", ")}"))
