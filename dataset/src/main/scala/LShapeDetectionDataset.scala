package dataset

import dimwit.*
import dimwit.python.PyBridge.liftPyTensor
import dimwit.python.PyBridge.liftPyTensor1
import dimwit.tensor.Tensor4
import me.shadaj.scalapy.py
import me.shadaj.scalapy.py.SeqConverters

import java.nio.file.Files
import java.nio.file.StandardCopyOption

/** The objects in an image: a [[Box]] per slot, labelled with an [[ObjectClass.id]]. */
final case class Detection[Slot, V](
    box: Box[Tuple1[Slot], V],
    label: Tensor1[Slot, Int32]
)

/** [[Detection]] for a batch of images along the axis `S`. */
final case class DetectionBatch[S, Slot, V](
    box: Box[(S, Slot), V],
    label: Tensor2[S, Slot, Int32]
)

/** One drawing together with the objects to detect in it. */
final case class LShapeSample[W, H, C, Slot](
    image: Tensor3[W, H, C, Float32],
    objects: Detection[Slot, Float32]
)

/** A batch of drawings together with the objects to detect in them. */
final case class LShapeBatch[S, W, H, C, Slot](
    images: Tensor4[S, W, H, C, Float32],
    objects: DetectionBatch[S, Slot, Float32]
)

/** [[NoObject]] is DETR's "no object" class and marks an unused query slot. */
enum ObjectClass(val id: Int):
  case NoObject extends ObjectClass(0)
  case PartLine extends ObjectClass(1)
  case Text extends ObjectClass(2)

object ObjectClass:

  def fromId(id: Int): ObjectClass =
    values.find(_.id == id).getOrElse(throw IllegalArgumentException(s"unknown object class id: $id"))

  /** The drawing action each detected class is rendered from. */
  private[dataset] val actionTypes: Map[ObjectClass, String] = Map(
    PartLine -> "PartLineWithId",
    Text -> "AnnotationTextRefId"
  )

/** DimWit wrapper around the [[https://huggingface.co/datasets/benikm91/l-shape benikm91/l-shape]]
  * dataset, backed by ScalaPy.
  *
  * The dataset ships its splits as plain repository files (`{split}_images.npy`,
  * `{split}_labels.jsonl`) rather than as a `datasets` config, so the Python side
  * (`python/l_shape_dataset.py`) fetches them with `huggingface_hub`, memory maps the
  * images with `numpy` and turns the drawing programs into detection targets. Only the
  * requested samples are ever read, which matters because the train split is ~8.6 GB.
  *
  * A drawing program is a graph of element nodes and relationship edges, of which only
  * the nodes that are drawn as objects are detected:
  *
  *   - [[ObjectClass.PartLine]] — a straight segment of the l-shape outline. Its box
  *     spans the two end points, widened to [[LShapeDetectionDataset.Config.minObjectSizePixels]]
  *     in the degenerate direction, since every line is axis aligned.
  *   - [[ObjectClass.Text]] — a dimension annotation. The dataset only gives its anchor
  *     point, so the box is a fixed [[LShapeDetectionDataset.Config.textBoxSizePixels]]
  *     square centred on it.
  *
  * Everything else (`ConnectTwoElementsWithId` relations, the `HelpLine` and
  * `BothSidedArrow` decorations of an annotation, `FinishDrawing`) is dropped. Targets
  * are padded to [[LShapeDetectionDataset.numQueries]] slots with [[ObjectClass.NoObject]],
  * so samples and batches are rectangular.
  *
  * Call `dimwit.initialize()` once before using this loader — it configures the Python
  * interpreter ScalaPy talks to.
  *
  * {{{
  * dimwit.initialize()
  *
  * trait Width derives Label
  * trait Height derives Label
  * trait Channel derives Label
  * trait BoundingBox derives Label
  * trait Batch derives Label
  *
  * val data = LShapeDetectionDataset.open(Axis[Width], Axis[Height], Axis[Channel], Axis[BoundingBox])(Split.Validation)
  * val sample = data.samples().next()
  * val batch = data.batches(Axis[Batch] -> 16).next()
  * }}}
  */
object LShapeDetectionDataset:

  /** A split of the dataset, named after its file prefix in the repository. */
  enum Split(val fileName: String):
    case Train extends Split("train")
    case Validation extends Split("val")

  /** @param numQueries      Number of target slots per sample. `None` uses the largest
    *                        number of objects occurring in the split; pin it when
    *                        training across splits, since a smaller value truncates
    *                        (reported by [[LShapeDetectionDataset.truncatedObjects]]).
    * @param minObjectSizePixels Minimum box extent, so that an axis aligned line gets a
    *                        non-degenerate box.
    * @param textBoxSizePixels Box extent of a text annotation, which the dataset locates
    *                        by a single anchor point.
    * @param normalizeImages Scale pixels from `[0, 255]` to `[0, 1]`. The renderings are
    *                        dark ink on a white background, so blank canvas is 1.0.
    * @param cacheLabels     Cache the parsed targets on disk (under
    *                        `$XDG_CACHE_HOME/dimwit-l-shape`, or `LSHAPE_CACHE_DIR`) so
    *                        the 131k line JSONL is parsed once.
    */
  final case class Config(
      repoId: String = "benikm91/l-shape",
      revision: Option[String] = None,
      numQueries: Option[Int] = None,
      minObjectSizePixels: Double = 4.0,
      textBoxSizePixels: Double = 12.0,
      normalizeImages: Boolean = true,
      cacheLabels: Boolean = true
  )

  /** Opens a split, downloading the repository files on first use (they are kept in the
    * HuggingFace cache afterwards).
    *
    * The axes name what the loaded tensors are labelled with: `width` and `height` the
    * image directions, `channel` its single greyscale channel and `box` the target slots.
    */
  def open[W: Label, H: Label, C: Label, Slot: Label](
      width: Axis[W],
      height: Axis[H],
      channel: Axis[C],
      slot: Axis[Slot]
  )(split: Split, config: Config = Config()): LShapeDetectionDataset[W, H, C, Slot] =
    require(config.numQueries.forall(_ > 0), "numQueries must be positive")
    val detected = ObjectClass.actionTypes.toSeq
    val handle = module.open_split(
      config.repoId,
      split.fileName,
      config.revision.getOrElse(""),
      detected.map((_, actionType) => actionType).toPythonCopy,
      detected.map((objectClass, _) => objectClass.id).toPythonCopy,
      config.numQueries.getOrElse(0),
      config.minObjectSizePixels,
      config.textBoxSizePixels,
      config.normalizeImages,
      config.cacheLabels
    )
    new LShapeDetectionDataset(handle, split, config)

  private val ModuleName = "l_shape_dataset"

  /** Touching `Jax.np` first makes sure DimWit has configured the interpreter and
    * `sys.path` before any Python object of ours is created.
    */
  private lazy val module: py.Dynamic =
    dimwit.jax.Jax.np
    val sys = py.module("sys")
    Option(getClass.getResourceAsStream(s"/python/$ModuleName.py")) match
      case Some(stream) =>
        try
          val directory = Files.createTempDirectory("l-shape-python")
          Files.copy(stream, directory.resolve(s"$ModuleName.py"), StandardCopyOption.REPLACE_EXISTING)
          sys.path.append(directory.toAbsolutePath.toString)
          Runtime.getRuntime.addShutdownHook(new Thread(() =>
            try
              Files.walk(directory).sorted(java.util.Comparator.reverseOrder()).forEach(Files.delete)
            catch case _: Exception => () // best effort cleanup
          ))
        finally stream.close()
      case None =>
        sys.path.append("./dataset/src/main/resources/python")
    py.module(ModuleName)

/** A single split of the l-shape dataset. Use [[LShapeDetectionDataset.open]] to create one. */
final class LShapeDetectionDataset[W: Label, H: Label, C: Label, Slot: Label] private[dataset] (
    private val handle: py.Dynamic,
    val split: LShapeDetectionDataset.Split,
    val config: LShapeDetectionDataset.Config
):

  val numSamples: Int = handle.num_samples.as[Int]
  val imageWidth: Int = handle.image_width.as[Int]
  val imageHeight: Int = handle.image_height.as[Int]

  /** Number of target slots per sample, i.e. the extent of the box axis. */
  val numQueries: Int = handle.num_queries.as[Int]

  /** Largest number of objects found in a sample of this split. */
  val maxObjects: Int = handle.observed_max_objects.as[Int]

  /** Number of objects dropped because [[numQueries]] was too small. */
  val truncatedObjects: Int = handle.truncated_objects.as[Int]

  /** The whole split, one sample at a time.
    *
    * @param shuffle draws the samples in a random order instead of storage order. Reusing
    *                the same `Random` across passes gives a fresh order each time.
    */
  def samples(shuffle: Option[scala.util.Random] = None): Iterator[LShapeSample[W, H, C, Slot]] =
    readOrder(shuffle).iterator.map: index =>
      val sample = handle.sample(index)
      LShapeSample(
        image = liftPyTensor[(W, H, C), Float32](sample.images),
        objects = Detection(
          box = Box(
            centerX = liftPyTensor1(Axis[Slot], VType[Float32])(sample.center_x),
            centerY = liftPyTensor1(Axis[Slot], VType[Float32])(sample.center_y),
            width = liftPyTensor1(Axis[Slot], VType[Float32])(sample.width),
            height = liftPyTensor1(Axis[Slot], VType[Float32])(sample.height)
          ),
          label = liftPyTensor1(Axis[Slot], VType[Int32])(sample.label)
        )
      )

  /** The whole split, one batch at a time. The incomplete tail is dropped, so every batch
    * has exactly `batchExtent.size` samples.
    *
    * @param shuffle see [[samples]].
    */
  def batches[S: Label](
      batchExtent: AxisExtent[S],
      shuffle: Option[scala.util.Random] = None
  ): Iterator[LShapeBatch[S, W, H, C, Slot]] =
    val batchSize = batchExtent.size
    require(batchSize > 0, "batch size must be positive")
    require(
      batchSize <= numSamples,
      s"batch size $batchSize exceeds the $numSamples samples of the ${split.fileName} split"
    )
    readOrder(shuffle).grouped(batchSize).filter(_.size == batchSize).map: indices =>
      val batch = handle.batch(indices.toPythonCopy)
      LShapeBatch(
        images = liftPyTensor[(S, W, H, C), Float32](batch.images),
        objects = DetectionBatch(
          box = Box(
            centerX = liftPyTensor[(S, Slot), Float32](batch.center_x),
            centerY = liftPyTensor[(S, Slot), Float32](batch.center_y),
            width = liftPyTensor[(S, Slot), Float32](batch.width),
            height = liftPyTensor[(S, Slot), Float32](batch.height)
          ),
          label = liftPyTensor[(S, Slot), Int32](batch.label)
        )
      )

  private def readOrder(shuffle: Option[scala.util.Random]): Seq[Int] =
    val indices = 0 until numSamples
    shuffle.fold(indices)(_.shuffle(indices))

  override def toString: String =
    s"LShapeDetectionDataset(${config.repoId}, ${split.fileName}, samples=$numSamples, " +
      s"image=${imageWidth}x${imageHeight}, queries=$numQueries)"
