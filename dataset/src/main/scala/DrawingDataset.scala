package dataset

import dimwit.*
import dimwit.Conversions.given
import dimwit.jax.Jax
import dimwit.python.PyBridge.liftPyTensor
import dimwit.tensor.Tensor4
import me.shadaj.scalapy.py

import scala.language.implicitConversions

/** The side of a drawing, in pixels. Every corpus renders onto the same canvas. */
val Canvas = 256

/** One drawing and what is to be predicted in it. */
final case class Sample[W, H, C, Target](image: Tensor3[W, H, C, Float32], target: Target):
  def map[T](f: Target => T): Sample[W, H, C, T] = Sample(image, f(target))

/** A batch of drawings and what is to be predicted in them. */
final case class Batch[S, W, H, C, Target](images: Tensor4[S, W, H, C, Float32], target: Target):
  def map[T](f: Target => T): Batch[S, W, H, C, T] = Batch(images, f(target))

/** Axis of the drawings of a split. */
private trait Drawings derives Label

/** A corpus of drawings, and how much room a record of it needs.
  *
  * The corpora differ in what they draw and in nothing else: the same drawing vocabulary, the same
  * canvas, the same files. So the only thing a corpus has to carry beyond where it is published is
  * how many nodes and relationships the largest record of it holds — which is what every slot in
  * this codebase is sized from, and the one number that must not be guessed. Both are measured
  * over the training split, which is the wider of the two.
  */
enum Corpus(val repoId: String, val maxNodes: Int, val maxEdges: Int):

  /** Six lines forming an L, and up to six annotations of them. */
  case LShape extends Corpus("benikm91/l-shape", 12, 12)

  /** A general rectilinear part of six to eighteen lines, and up to that many annotations. */
  case Rectilinear6to18 extends Corpus("benikm91/rectilinear-6to18", 22, 22)

/** DimWit wrapper around the drawing datasets, backed by ScalaPy.
  *
  * [[DrawingDataset.samples]] and [[DrawingDataset.batches]] hand out the [[Record]] every drawing
  * was rendered from. [[DrawingDataset.objects]] and [[DrawingDataset.objectBatches]] hand out the
  * same drawings as something to detect, which is that very data through [[Objects.of]] and
  * nothing else.
  *
  * Call `dimwit.initialize()` once before using this loader.
  *
  * {{{
  * dimwit.initialize()
  *
  * val data = DrawingDataset.open(Corpus.LShape)(Axis[Width], Axis[Height], Axis[Channel], Axis[Node], Axis[Edge])(Split.Train)
  * val record = data.samples.next().target
  * val detected = data.objectBatches(Axis[Drawings] -> 16).next().target
  * }}}
  */
object DrawingDataset:

  enum Split(val fileName: String):
    case Train extends Split("train")
    case Validation extends Split("val")

  /** Opens a split of a corpus, downloading the repository files on first use.
    *
    * The records are read into tensors here and for good; the drawings stay memory mapped, since
    * a train split runs to gigabytes and only the batches asked for are ever read.
    */
  def open[W: Label, H: Label, C: Label, Node: Label, Edge: Label](corpus: Corpus)(
      width: Axis[W],
      height: Axis[H],
      channel: Axis[C],
      node: Axis[Node],
      edge: Axis[Edge]
  )(split: Split): DrawingDataset[W, H, C, Node, Edge] =
    val parsed = module.records(
      corpus.repoId,
      split.fileName,
      corpus.maxNodes,
      corpus.maxEdges,
      NodeClass.NoNode.id,
      NodeClass.Line.id,
      NodeClass.Annotation.id,
      EdgeClass.NoEdge.id,
      EdgeClass.Connected.id,
      EdgeClass.Annotates.id
    )
    def read(at: Int) = Jax.jnp.asarray(parsed.applyDynamic("__getitem__")(at))
    new DrawingDataset(
      corpus,
      module.drawings(corpus.repoId, split.fileName),
      RecordBatch(
        nodeClass = liftPyTensor[(Drawings, Node), Int32](read(0)),
        startX = liftPyTensor[(Drawings, Node), Float32](read(1)),
        startY = liftPyTensor[(Drawings, Node), Float32](read(2)),
        endX = liftPyTensor[(Drawings, Node), Float32](read(3)),
        endY = liftPyTensor[(Drawings, Node), Float32](read(4)),
        edgeClass = liftPyTensor[(Drawings, Edge), Int32](read(5)),
        subject = liftPyTensor[(Drawings, Edge), Int32](read(6)),
        obj = liftPyTensor[(Drawings, Edge), Int32](read(7))
      )
    )

  private lazy val module: py.Dynamic = PythonModules("drawing_dataset")

/** A single split of a drawing dataset. Use [[DrawingDataset.open]] to create one. */
final class DrawingDataset[W: Label, H: Label, C: Label, Node: Label, Edge: Label] private[dataset] (
    val corpus: Corpus,
    private val images: py.Dynamic,
    private val records: RecordBatch[Drawings, Node, Edge]
):

  val numSamples: Int = records.nodeClass.shape(Axis[Drawings])

  /** Every drawing of the split, once. */
  def samples: Iterator[Sample[W, H, C, Record[Node, Edge]]] =
    (0 until numSamples).iterator.map: at =>
      Sample(drawn(Axis[Drawings] -> 1, at).slice(Axis[Drawings].at(0)), recordAt(at))

  /** Batches of drawings, for as long as they are asked for. The drawings were generated
    * independently of one another, so reading them in order is already a shuffle.
    */
  def batches[S: Label](batch: AxisExtent[S]): Iterator[Batch[S, W, H, C, RecordBatch[S, Node, Edge]]] =
    require(batch.size <= numSamples, s"a batch of ${batch.size} exceeds the $numSamples drawings of the split")
    val starts = 0 to numSamples - batch.size by batch.size
    Iterator.continually(starts).flatten.map(from => Batch(drawn(batch, from), recordsIn(batch, from)))

  /** The same drawings as something to detect. */
  def objects: Iterator[Sample[W, H, C, Objects[Node]]] = samples.map(_.map(Objects.of))

  def objectBatches[S: Label](batch: AxisExtent[S]): Iterator[Batch[S, W, H, C, ObjectBatch[S, Node]]] =
    batches(batch).map(_.map(Objects.of))

  override def toString: String = s"DrawingDataset(${corpus.repoId}, drawings=$numSamples, nodes=${corpus.maxNodes})"

  /** The drawings from `from` on, as ink on a white canvas in `[0, 1]`. They are stored row major
    * — row index = y — so the axes are swapped to put x first, as a record's coordinates are.
    */
  private def drawn[S: Label](rows: AxisExtent[S], from: Int): Tensor4[S, W, H, C, Float32] =
    val pixels = images.applyDynamic("__getitem__")(py.Dynamic.global.slice(from, from + rows.size))
    liftPyTensor[(S, H, W), UInt8](Jax.jnp.asarray(pixels))
      .swap(Axis[H], Axis[W])
      .appendAxis(Axis[C])
      .asFloat(VType[Float32]) /! 255f

  private def recordAt(at: Int): Record[Node, Edge] =
    val drawing = Axis[Drawings].at(at)
    Record(
      records.nodeClass.slice(drawing),
      records.startX.slice(drawing),
      records.startY.slice(drawing),
      records.endX.slice(drawing),
      records.endY.slice(drawing),
      records.edgeClass.slice(drawing),
      records.subject.slice(drawing),
      records.obj.slice(drawing)
    )

  private def recordsIn[S: Label](rows: AxisExtent[S], from: Int): RecordBatch[S, Node, Edge] =
    val taken = Axis[Drawings].at(from until from + rows.size)
    RecordBatch(
      records.nodeClass.slice(taken).relabel(Axis[Drawings] -> rows.axis),
      records.startX.slice(taken).relabel(Axis[Drawings] -> rows.axis),
      records.startY.slice(taken).relabel(Axis[Drawings] -> rows.axis),
      records.endX.slice(taken).relabel(Axis[Drawings] -> rows.axis),
      records.endY.slice(taken).relabel(Axis[Drawings] -> rows.axis),
      records.edgeClass.slice(taken).relabel(Axis[Drawings] -> rows.axis),
      records.subject.slice(taken).relabel(Axis[Drawings] -> rows.axis),
      records.obj.slice(taken).relabel(Axis[Drawings] -> rows.axis)
    )
