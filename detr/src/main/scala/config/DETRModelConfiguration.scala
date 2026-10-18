package detr.config

/** How big a model a run asks for, which is the one thing about the model a run gets to choose.
  * Everything else is fixed, so that two runs of the same size are the same model.
  *
  * `s-deep` is the transformer of [[https://arxiv.org/abs/2005.12872 DETR]] itself: six layers
  * either side, 256 wide, eight heads.
  */
enum DETRModelConfiguration(val name: String, val embedding: Int, val numLayers: Int, val numHeads: Int):
  case XS extends DETRModelConfiguration("xs", 128, 3, 4)
  case S extends DETRModelConfiguration("s", 256, 3, 8)
  case SDeep extends DETRModelConfiguration("s-deep", 256, 6, 8)

object DETRModelConfiguration:

  /** The size a run names on its command line. */
  def named(name: String): DETRModelConfiguration =
    values.find(_.name == name).getOrElse(sys.error(s"no size named '$name': ${values.map(_.name).mkString(", ")}"))
