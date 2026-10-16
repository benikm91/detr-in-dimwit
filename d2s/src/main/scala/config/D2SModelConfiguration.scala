package d2s.config

// Model sizes for experiments
enum D2SModelConfiguration(val name: String, val embedding: Int, val numLayers: Int, val numHeads: Int):
  case XS extends D2SModelConfiguration("xs", 128, 3, 4)
  case S extends D2SModelConfiguration("s", 256, 3, 8)
  case SDeep extends D2SModelConfiguration("s-deep", 256, 6, 8)
  case MDeep extends D2SModelConfiguration("m-deep", 512, 6, 8)

object D2SModelConfiguration:
  def named(name: String): D2SModelConfiguration = values.find(_.name == name).getOrElse(sys.error(s"no size named '$name': ${values.map(_.name).mkString(", ")}"))
