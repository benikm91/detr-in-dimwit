package dataset

import dimwit.*
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

/** A batch of drawings and what is to be predicted in them, both still on the host as the corpus
  * stores them, so that sharding the batch is its one transfer to the devices. [[drawingsOf]]
  * turns the pixels into what a model reads.
  */
final case class Batch[S, W, H, Target](pixels: Tensor3[S, H, W, UInt8], target: Target):
  def map[T](f: Target => T): Batch[S, W, H, T] = Batch(pixels, f(target))

/** Drawings as a corpus stores them — rows first, one byte a pixel — as a model reads them: x
  * first, with a channel axis, ink on a white canvas in `[0, 1]`. Cheap enough to run inside a
  * jitted step, which is where a training batch is turned.
  */
def drawingsOf[S: Label, W: Label, H: Label, C: Label](pixels: Tensor3[S, H, W, UInt8], channel: Axis[C]): Tensor4[S, W, H, C, Float32] =
  pixels.swap(Axis[H], Axis[W]).appendAxis(channel).asFloat(VType[Float32]) /! 255f

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
enum Corpus(val name: String, val repoId: String, val maxNodes: Int, val maxEdges: Int):

  /** Six lines forming an L, and up to six annotations of them. */
  case LShape extends Corpus("l-shape", "benikm91/l-shape", 12, 12)

  /** A general rectilinear part of six to eighteen lines, and up to that many annotations. */
  case Rectilinear6to18 extends Corpus("rectilinear", "benikm91/rectilinear-6to18", 22, 22)

  /** Five to sixteen lines and circles of a CAD sketch, drawn from
    * [[https://sketchgraphs.cs.princeton.edu SketchGraphs]]. The sketches are kept for what they
    * draw, not for how they are constrained, so no relationship is held between them.
    */
  case SketchGraph extends Corpus("sketch", "benikm91/sketch-graph", 16, 0)

  /** The same sketches with arcs too, and every sketch of SketchGraphs that is drawn in lines,
    * circles and arcs alone rather than a sample of them.
    */
  case SketchGraphXL extends Corpus("sketch-xl", "benikm91/sketch-graph-xl", 16, 0)

object Corpus:

  /** The corpus a run names on its command line. */
  def named(name: String): Corpus =
    values.find(_.name == name).getOrElse(sys.error(s"no corpus named '$name': ${values.map(_.name).mkString(", ")}"))

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
      NodeClass.Circle.id,
      NodeClass.Arc.id,
      EdgeClass.NoEdge.id,
      EdgeClass.Connected.id,
      EdgeClass.Annotates.id
    )
    def read(at: Int) = Jax.jnp.asarray(parsed.applyDynamic("__getitem__")(at))
    new DrawingDataset(
      corpus,
      module.drawings(corpus.repoId, split.fileName),
      parsed,
      RecordBatch(
        nodeClass = liftPyTensor[(Drawings, Node), Int32](read(0)),
        startX = liftPyTensor[(Drawings, Node), Float32](read(1)),
        startY = liftPyTensor[(Drawings, Node), Float32](read(2)),
        endX = liftPyTensor[(Drawings, Node), Float32](read(3)),
        endY = liftPyTensor[(Drawings, Node), Float32](read(4)),
        midX = liftPyTensor[(Drawings, Node), Float32](read(5)),
        midY = liftPyTensor[(Drawings, Node), Float32](read(6)),
        edgeClass = liftPyTensor[(Drawings, Edge), Int32](read(7)),
        subject = liftPyTensor[(Drawings, Edge), Int32](read(8)),
        obj = liftPyTensor[(Drawings, Edge), Int32](read(9))
      )
    )

  private lazy val module: py.Dynamic = PythonModules("drawing_dataset")

/** A single split of a drawing dataset. Use [[DrawingDataset.open]] to create one. */
final class DrawingDataset[W: Label, H: Label, C: Label, Node: Label, Edge: Label] private[dataset] (
    val corpus: Corpus,
    private val images: py.Dynamic,
    /** The records as parsed, on the host, which is where a training batch is cut from. */
    private val parsed: py.Dynamic,
    private val records: RecordBatch[Drawings, Node, Edge]
):

  val numSamples: Int = records.nodeClass.shape(Axis[Drawings])

  /** Every drawing of the split, once. */
  def samples: Iterator[Sample[W, H, C, Record[Node, Edge]]] =
    (0 until numSamples).iterator.map: at =>
      Sample(drawn(Axis[Drawings] -> 1, at).slice(Axis[Drawings].at(0)), recordAt(at))

  /** Batches of drawings, for as long as they are asked for, on the host. The drawings were
    * generated independently of one another, so reading them in order is already a shuffle.
    */
  def batches[S: Label](batch: AxisExtent[S]): Iterator[Batch[S, W, H, RecordBatch[S, Node, Edge]]] =
    require(batch.size <= numSamples, s"a batch of ${batch.size} exceeds the $numSamples drawings of the split")
    val starts = 0 to numSamples - batch.size by batch.size
    Iterator.continually(starts).flatten.map(from => Batch(stored(batch, from), recordsIn(batch, from)))

  /** The same drawings as something to detect. */
  def objects: Iterator[Sample[W, H, C, Objects[Node]]] = samples.map(_.map(Objects.of))

  def objectBatches[S: Label](batch: AxisExtent[S]): Iterator[Batch[S, W, H, ObjectBatch[S, Node]]] =
    batches(batch).map(_.map(Objects.of))

  override def toString: String = s"DrawingDataset(${corpus.repoId}, drawings=$numSamples, nodes=${corpus.maxNodes})"

  /** The drawings from `from` on, as stored — row major, row index = y — and still on the host. */
  private def stored[S: Label](rows: AxisExtent[S], from: Int): Tensor3[S, H, W, UInt8] =
    liftPyTensor[(S, H, W), UInt8](images.applyDynamic("__getitem__")(py.Dynamic.global.slice(from, from + rows.size)))

  /** The drawings from `from` on, on the device, as a model reads them. */
  private def drawn[S: Label](rows: AxisExtent[S], from: Int): Tensor4[S, W, H, C, Float32] =
    val pixels = images.applyDynamic("__getitem__")(py.Dynamic.global.slice(from, from + rows.size))
    drawingsOf(liftPyTensor[(S, H, W), UInt8](Jax.jnp.asarray(pixels)), Axis[C])

  private def recordAt(at: Int): Record[Node, Edge] =
    val drawing = Axis[Drawings].at(at)
    Record(
      records.nodeClass.slice(drawing),
      records.startX.slice(drawing),
      records.startY.slice(drawing),
      records.endX.slice(drawing),
      records.endY.slice(drawing),
      records.midX.slice(drawing),
      records.midY.slice(drawing),
      records.edgeClass.slice(drawing),
      records.subject.slice(drawing),
      records.obj.slice(drawing)
    )

  /** The records of the drawings from `from` on, cut from the parsed ones and still on the host. */
  private def recordsIn[S: Label](rows: AxisExtent[S], from: Int): RecordBatch[S, Node, Edge] =
    val taken = py.Dynamic.global.slice(from, from + rows.size)
    def cut(at: Int) = parsed.applyDynamic("__getitem__")(at).applyDynamic("__getitem__")(taken)
    RecordBatch(
      nodeClass = liftPyTensor[(S, Node), Int32](cut(0)),
      startX = liftPyTensor[(S, Node), Float32](cut(1)),
      startY = liftPyTensor[(S, Node), Float32](cut(2)),
      endX = liftPyTensor[(S, Node), Float32](cut(3)),
      endY = liftPyTensor[(S, Node), Float32](cut(4)),
      midX = liftPyTensor[(S, Node), Float32](cut(5)),
      midY = liftPyTensor[(S, Node), Float32](cut(6)),
      edgeClass = liftPyTensor[(S, Edge), Int32](cut(7)),
      subject = liftPyTensor[(S, Edge), Int32](cut(8)),
      obj = liftPyTensor[(S, Edge), Int32](cut(9))
    )
