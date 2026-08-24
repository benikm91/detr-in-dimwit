package dataset

import dimwit.*
import dimwit.Conversions.given
import dimwit.jax.Jax
import dimwit.python.PyBridge.liftPyTensor
import dimwit.tensor.Tensor4
import me.shadaj.scalapy.py

import java.nio.file.Files
import java.nio.file.StandardCopyOption
import scala.language.implicitConversions

/** The side of a drawing, in pixels. */
val Canvas = 256

/** The most nodes a record of this dataset holds: six lines, up to six annotations, six corners
  * and up to six annotated lines.
  */
val MaxNodes = 24

/** One drawing and what is to be predicted in it. */
final case class Sample[W, H, C, Target](image: Tensor3[W, H, C, Float32], target: Target):
  def map[T](f: Target => T): Sample[W, H, C, T] = Sample(image, f(target))

/** A batch of drawings and what is to be predicted in them. */
final case class Batch[S, W, H, C, Target](images: Tensor4[S, W, H, C, Float32], target: Target):
  def map[T](f: Target => T): Batch[S, W, H, C, T] = Batch(images, f(target))

/** Axis of the drawings of a split. */
private trait Drawings derives Label

/** DimWit wrapper around the [[https://huggingface.co/datasets/benikm91/l-shape benikm91/l-shape]]
  * dataset, backed by ScalaPy.
  *
  * [[LShapeDataset.samples]] and [[LShapeDataset.batches]] hand out the [[Record]] every drawing
  * was rendered from. [[LShapeDataset.objects]] and [[LShapeDataset.objectBatches]] hand out the
  * same drawings as something to detect, which is that very data through [[Objects.of]] and
  * nothing else.
  *
  * Call `dimwit.initialize()` once before using this loader.
  *
  * {{{
  * dimwit.initialize()
  *
  * val data = LShapeDataset.open(Axis[Width], Axis[Height], Axis[Channel], Axis[Node])(Split.Train)
  * val record = data.samples.next().target
  * val detected = data.objectBatches(Axis[Drawings] -> 16).next().target
  * }}}
  */
object LShapeDataset:

  enum Split(val fileName: String):
    case Train extends Split("train")
    case Validation extends Split("val")

  /** Opens a split, downloading the repository files on first use.
    *
    * The records are read into tensors here and for good; the drawings stay memory mapped, since
    * the train split is 8.6 GB and only the batches asked for are ever read.
    */
  def open[W: Label, H: Label, C: Label, Node: Label](
      width: Axis[W],
      height: Axis[H],
      channel: Axis[C],
      node: Axis[Node]
  )(split: Split): LShapeDataset[W, H, C, Node] =
    val parsed = module.records(
      split.fileName,
      MaxNodes,
      NodeClass.NoNode.id,
      NodeClass.Line.id,
      NodeClass.Annotation.id,
      NodeClass.Connected.id,
      NodeClass.Annotates.id
    )
    def read(at: Int) = Jax.jnp.asarray(parsed.applyDynamic("__getitem__")(at))
    new LShapeDataset(
      module.drawings(split.fileName),
      RecordBatch(
        nodeClass = liftPyTensor[(Drawings, Node), Int32](read(0)),
        xs = liftPyTensor[(Drawings, Node, NodePoint), Float32](read(1)),
        ys = liftPyTensor[(Drawings, Node, NodePoint), Float32](read(2)),
        links = liftPyTensor[(Drawings, Node, NodeLink), Int32](read(3))
      )
    )

  /** Touching `Jax.np` first makes sure DimWit has configured the interpreter and `sys.path`
    * before any Python object of ours is created.
    */
  private lazy val module: py.Dynamic =
    Jax.np
    val sys = py.module("sys")
    Option(getClass.getResourceAsStream("/python/l_shape_dataset.py")) match
      case Some(stream) =>
        try
          val directory = Files.createTempDirectory("l-shape-python")
          Files.copy(stream, directory.resolve("l_shape_dataset.py"), StandardCopyOption.REPLACE_EXISTING)
          sys.path.append(directory.toAbsolutePath.toString)
          Runtime.getRuntime.addShutdownHook(new Thread(() =>
            try Files.walk(directory).sorted(java.util.Comparator.reverseOrder()).forEach(Files.delete)
            catch case _: Exception => () // best effort cleanup
          ))
        finally stream.close()
      case None =>
        sys.path.append("./dataset/src/main/resources/python")
    py.module("l_shape_dataset")

/** A single split of the l-shape dataset. Use [[LShapeDataset.open]] to create one. */
final class LShapeDataset[W: Label, H: Label, C: Label, Node: Label] private[dataset] (
    private val images: py.Dynamic,
    private val records: RecordBatch[Drawings, Node]
):

  val numSamples: Int = records.nodeClass.shape(Axis[Drawings])

  /** Every drawing of the split, once. */
  def samples: Iterator[Sample[W, H, C, Record[Node]]] =
    (0 until numSamples).iterator.map: at =>
      Sample(drawn(Axis[Drawings] -> 1, at).slice(Axis[Drawings].at(0)), recordAt(at))

  /** Batches of drawings, for as long as they are asked for. The drawings were generated
    * independently of one another, so reading them in order is already a shuffle.
    */
  def batches[S: Label](batch: AxisExtent[S]): Iterator[Batch[S, W, H, C, RecordBatch[S, Node]]] =
    require(batch.size <= numSamples, s"a batch of ${batch.size} exceeds the $numSamples drawings of the split")
    val starts = 0 to numSamples - batch.size by batch.size
    Iterator.continually(starts).flatten.map(from => Batch(drawn(batch, from), recordsIn(batch, from)))

  /** The same drawings as something to detect. */
  def objects: Iterator[Sample[W, H, C, Objects[Node]]] = samples.map(_.map(Objects.of))

  def objectBatches[S: Label](batch: AxisExtent[S]): Iterator[Batch[S, W, H, C, ObjectBatch[S, Node]]] =
    batches(batch).map(_.map(Objects.of))

  override def toString: String = s"LShapeDataset(drawings=$numSamples, nodes=$MaxNodes)"

  /** The drawings from `from` on, as ink on a white canvas in `[0, 1]`. They are stored row major
    * — row index = y — so the axes are swapped to put x first, as a record's coordinates are.
    */
  private def drawn[S: Label](rows: AxisExtent[S], from: Int): Tensor4[S, W, H, C, Float32] =
    val pixels = images.applyDynamic("__getitem__")(py.Dynamic.global.slice(from, from + rows.size))
    liftPyTensor[(S, H, W), UInt8](Jax.jnp.asarray(pixels))
      .swap(Axis[H], Axis[W])
      .appendAxis(Axis[C])
      .asFloat(VType[Float32]) /! 255f

  private def recordAt(at: Int): Record[Node] =
    val drawing = Axis[Drawings].at(at)
    Record(
      records.nodeClass.slice(drawing),
      records.xs.slice(drawing),
      records.ys.slice(drawing),
      records.links.slice(drawing)
    )

  private def recordsIn[S: Label](rows: AxisExtent[S], from: Int): RecordBatch[S, Node] =
    val taken = Axis[Drawings].at(from until from + rows.size)
    RecordBatch(
      records.nodeClass.slice(taken).relabel(Axis[Drawings] -> rows.axis),
      records.xs.slice(taken).relabel(Axis[Drawings] -> rows.axis),
      records.ys.slice(taken).relabel(Axis[Drawings] -> rows.axis),
      records.links.slice(taken).relabel(Axis[Drawings] -> rows.axis)
    )
