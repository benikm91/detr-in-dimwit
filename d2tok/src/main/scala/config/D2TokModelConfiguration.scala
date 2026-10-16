package d2tok.config

// Model sizes for experiments
enum D2TokModelConfiguration(val name: String, val embedding: Int, val numLayers: Int, val numHeads: Int):
  case XS extends D2TokModelConfiguration("xs", 128, 3, 4)
  case S extends D2TokModelConfiguration("s", 256, 3, 8)
  case SDeep extends D2TokModelConfiguration("s-deep", 256, 6, 8)
  case MDeep extends D2TokModelConfiguration("m-deep", 512, 6, 8)

object D2TokModelConfiguration:
  def named(name: String): D2TokModelConfiguration = values.find(_.name == name).getOrElse(sys.error(s"no size named '$name': ${values.map(_.name).mkString(", ")}"))
