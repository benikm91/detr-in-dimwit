package dataset

import dimwit.*
import dimwit.python.PyBridge.liftPyTensor
import dimwit.python.PyBridge.liftPyTensor1
import dimwit.tensor.Tensor4
import me.shadaj.scalapy.py
import me.shadaj.scalapy.py.SeqConverters

import java.nio.file.Files
import java.nio.file.StandardCopyOption

/** One drawing and what is to be predicted in it. */
final case class Sample[W, H, C, Target](image: Tensor3[W, H, C, Float32], target: Target):
  def map[T](f: Target => T): Sample[W, H, C, T] = Sample(image, f(target))

/** A batch of drawings and what is to be predicted in them. */
final case class Batch[S, W, H, C, Target](images: Tensor4[S, W, H, C, Float32], target: Target):
  def map[T](f: Target => T): Batch[S, W, H, C, T] = Batch(images, f(target))

/** DimWit wrapper around the [[https://huggingface.co/datasets/benikm91/l-shape benikm91/l-shape]] dataset.
  *
  * The dataset ships its splits as plain repository files (`{split}_images.npy`,
  * `{split}_labels.jsonl`) rather than as a `datasets` config, so the Python side
  * (`python/l_shape_dataset.py`) fetches them with `huggingface_hub`, memory maps the images with
  * `numpy` and turns every drawing program into the [[Record]] it draws. Only the requested
  * samples are ever read, which matters because the train split is ~8.6 GB.
  *
  * [[LShapeDataset.samples]] and [[LShapeDataset.batches]] hand out those records.
  * [[LShapeDataset.objects]] and [[LShapeDataset.objectBatches]] hand out the same drawings as objects to detect ([[Objects.of]]).
  */
object LShapeDataset:

  /** A split of the dataset, named after its file prefix in the repository. */
  enum Split(val fileName: String):
    case Train extends Split("train")
    case Validation extends Split("val")

  /** @param maxNumNodes How many slots a record is padded to. `None` uses the longest record of
    *                    the split; pin it when training across splits, since a smaller value
    *                    truncates (reported by [[LShapeDataset.truncatedNodes]]).
    * @param minObjectSizePixels The extent a box is widened to where a line is degenerate.
    * @param textBoxSizePixels   The extent of the box around an annotation, which is a point.
    * @param normalizeImages     Scale pixels from `[0, 255]` to `[0, 1]`. The renderings are dark
    *                            ink on a white background, so blank canvas is 1.0.
    * @param cacheRecords        Cache the parsed records on disk (under `$XDG_CACHE_HOME/dimwit-l-shape`,
    *                            or `LSHAPE_CACHE_DIR`) so the 131k line JSONL is parsed once.
    */
  final case class Config(
      repoId: String = "benikm91/l-shape",
      revision: Option[String] = None,
      maxNumNodes: Option[Int] = None,
      minObjectSizePixels: Double = 4.0,
      textBoxSizePixels: Double = 12.0,
      normalizeImages: Boolean = true,
      cacheRecords: Boolean = true
  )

  /** Opens a split, downloading the repository files on first use. */
  def open[W: Label, H: Label, C: Label, Node: Label](
      width: Axis[W],
      height: Axis[H],
      channel: Axis[C],
      node: Axis[Node]
  )(split: Split, config: Config = Config()): LShapeDataset[W, H, C, Node] =
    require(config.maxNumNodes.forall(_ > 0), "maxNumNodes must be positive")
    val handle = module.open_split(
      config.repoId,
      split.fileName,
      config.revision.getOrElse(""),
      NodeClass.NoNode.id,
      NodeClass.Line.id,
      NodeClass.Annotation.id,
      NodeClass.Connected.id,
      NodeClass.Annotates.id,
      config.maxNumNodes.getOrElse(0),
      config.normalizeImages,
      config.cacheRecords
    )
    new LShapeDataset(handle, split, config)

  private val ModuleName = "l_shape_dataset"

  /** Touching `Jax.np` first makes sure DimWit has configured the interpreter and `sys.path`
    * before any Python object of ours is created.
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

/** A single split of the l-shape dataset. Use [[LShapeDataset.open]] to create one. */
final class LShapeDataset[W: Label, H: Label, C: Label, Node: Label] private[dataset] (
    private val handle: py.Dynamic,
    val split: LShapeDataset.Split,
    val config: LShapeDataset.Config
):

  val numSamples: Int = handle.num_samples.as[Int]
  val imageWidth: Int = handle.image_width.as[Int]
  val imageHeight: Int = handle.image_height.as[Int]

  /** How many nodes a record is laid out in, i.e. the extent of the node axis. */
  val maxNumNodes: Int = handle.max_num_nodes.as[Int]

  /** The most nodes a record of this split actually holds. */
  val observedMaxNodes: Int = handle.observed_max_nodes.as[Int]

  /** Nodes dropped because [[maxNumNodes]] was too small. */
  val truncatedNodes: Int = handle.truncated_nodes.as[Int]

  /** Links dropped because they named an element the record does not hold. Zero for both splits
    * as the dataset stands.
    */
  val unresolvedLinks: Int = handle.unresolved_links.as[Int]

  /** How a record of this split is drawn, in fractions of the square canvas. */
  val geometry: Geometry = Geometry(
    minimumSize = (config.minObjectSizePixels / imageWidth).toFloat,
    annotationSize = (config.textBoxSizePixels / imageWidth).toFloat
  )

  /** The whole split, one record at a time.
    *
    * @param shuffle draws the samples in a random order instead of storage order. Reusing the
    *                same `Random` across passes gives a fresh order each time.
    */
  def samples(shuffle: Option[scala.util.Random] = None): Iterator[Sample[W, H, C, Record[Node]]] =
    readOrder(shuffle).iterator.map: index =>
      val sample = handle.sample(index)
      Sample(
        image = liftPyTensor[(W, H, C), Float32](sample.images),
        target = Record(
          nodeClass = liftPyTensor1(Axis[Node], VType[Int32])(sample.node_class),
          xs = liftPyTensor[(Node, NodePoint), Float32](sample.xs),
          ys = liftPyTensor[(Node, NodePoint), Float32](sample.ys),
          links = liftPyTensor[(Node, NodeLink), Int32](sample.links)
        )
      )

  /** The whole split, one batch of records at a time. The incomplete tail is dropped. */
  def batches[S: Label](
      batchExtent: AxisExtent[S],
      shuffle: Option[scala.util.Random] = None
  ): Iterator[Batch[S, W, H, C, RecordBatch[S, Node]]] =
    val batchSize = batchExtent.size
    require(batchSize > 0, "batch size must be positive")
    require(
      batchSize <= numSamples,
      s"batch size $batchSize exceeds the $numSamples samples of the ${split.fileName} split"
    )
    readOrder(shuffle).grouped(batchSize).filter(_.size == batchSize).map: indices =>
      val batch = handle.batch(indices.toPythonCopy)
      Batch(
        images = liftPyTensor[(S, W, H, C), Float32](batch.images),
        target = RecordBatch(
          nodeClass = liftPyTensor[(S, Node), Int32](batch.node_class),
          xs = liftPyTensor[(S, Node, NodePoint), Float32](batch.xs),
          ys = liftPyTensor[(S, Node, NodePoint), Float32](batch.ys),
          links = liftPyTensor[(S, Node, NodeLink), Int32](batch.links)
        )
      )

  /** The same drawings as something to detect. */
  def objects(shuffle: Option[scala.util.Random] = None): Iterator[Sample[W, H, C, Objects[Node]]] =
    samples(shuffle).map(_.map(Objects.of(_, geometry)))

  def objectBatches[S: Label](
      batchExtent: AxisExtent[S],
      shuffle: Option[scala.util.Random] = None
  ): Iterator[Batch[S, W, H, C, ObjectBatch[S, Node]]] =
    batches(batchExtent, shuffle).map(_.map(Objects.of(_, geometry)))

  private def readOrder(shuffle: Option[scala.util.Random]): Seq[Int] =
    val indices = 0 until numSamples
    shuffle.fold(indices)(_.shuffle(indices))

  override def toString: String =
    s"LShapeDataset(${config.repoId}, ${split.fileName}, samples=$numSamples, " +
      s"image=${imageWidth}x${imageHeight}, nodes=$maxNumNodes)"
