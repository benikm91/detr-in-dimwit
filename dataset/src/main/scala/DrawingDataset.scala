package dataset

import dimwit.*
import dimwit.jax.Jax
import dimwit.python.PyBridge.liftPyTensor
import dimwit.tensor.Tensor4
import me.shadaj.scalapy.py

import scala.language.implicitConversions

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

/** A drawing as a model reads it, back as the 8 bit grey levels [[plotwit]] plots. */
def greyLevels[W: Label, H: Label, C: Label, V: IsFloating](image: Tensor3[W, H, C, V]): Tensor2[W, H, UInt8] =
  (image.squeeze(Axis[C]).asFloat32 *! 255f).asInt(VType[UInt8])

/** Axis of the drawings of a split. */
private trait Drawings derives Label

/** A corpus of drawings, and how much room a record of it needs.
  *
  * The corpora differ in what they draw and in little else: the same record, the same files. So a
  * corpus carries where it is published, the side of its drawings in pixels, and how many nodes
  * and relationships the largest record of it holds — which is what every slot in this codebase is
  * sized from, and the one number that must not be guessed. Both are measured over the training
  * split, which is the widest.
  */
enum Corpus(val name: String, val repoId: String, val maxNodes: Int, val maxEdges: Int, val canvas: Int = 256, val folder: String = ""):

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

  /** The sketches of SketchGraphs as [[https://arxiv.org/abs/2109.14124 Vitruvion]] selected them,
    * in Vitruvion's own renders and splits, which PICASSO and DAVINCI are measured on: lines,
    * circles, arcs and points, construction geometry among them. Their primitives alone — the
    * constraints the same files hold are not read.
    */
  case VitruvionPrimitives extends Corpus("vitruvion-primitives", "benikm91/sketch-graph-vitruvion", 16, 0, canvas = 128, folder = "records")

object Corpus:

  /** The corpus a run names on its command line. */
  def named(name: String): Corpus =
    values.find(_.name == name).getOrElse(sys.error(s"no corpus named '$name': ${values.map(_.name).mkString(", ")}"))

/** DimWit wrapper around the drawing datasets, backed by ScalaPy.
  *
  * [[DrawingDataset.samples]] and [[DrawingDataset.batches]] hand out the [[Record]] every drawing
  * was rendered from.
  *
  * Call `dimwit.initialize()` once before using this loader.
  *
  * {{{
  * dimwit.initialize()
  *
  * val data = DrawingDataset.open(Corpus.LShape)(Axis[Width], Axis[Height], Axis[Channel], Axis[Node], Axis[Edge])(Split.Train)
  * val record = data.samples.next().target
  * }}}
  */
object DrawingDataset:

  enum Split(val fileName: String):
    case Train extends Split("train")
    case Validation extends Split("val")
    case Test extends Split("test")

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
      corpus.folder,
      split.fileName,
      corpus.canvas,
      corpus.maxNodes,
      corpus.maxEdges,
      NodeClass.NoNode.id,
      NodeClass.Line.id,
      NodeClass.Annotation.id,
      NodeClass.Circle.id,
      NodeClass.Arc.id,
      NodeClass.Point.id,
      EdgeClass.NoEdge.id,
      EdgeClass.Connected.id,
      EdgeClass.Annotates.id
    )
    def read(at: Int) = Jax.jnp.asarray(parsed.applyDynamic("__getitem__")(at))
    new DrawingDataset(
      corpus,
      module.drawings(corpus.repoId, corpus.folder, split.fileName),
      parsed,
      RecordBatch(
        nodeClass = liftPyTensor[(Drawings, Node), Int32](read(0)),
        construction = liftPyTensor[(Drawings, Node), Int32](read(1)),
        startX = liftPyTensor[(Drawings, Node), Float32](read(2)),
        startY = liftPyTensor[(Drawings, Node), Float32](read(3)),
        endX = liftPyTensor[(Drawings, Node), Float32](read(4)),
        endY = liftPyTensor[(Drawings, Node), Float32](read(5)),
        midX = liftPyTensor[(Drawings, Node), Float32](read(6)),
        midY = liftPyTensor[(Drawings, Node), Float32](read(7)),
        edgeClass = liftPyTensor[(Drawings, Edge), Int32](read(8)),
        subject = liftPyTensor[(Drawings, Edge), Int32](read(9)),
        obj = liftPyTensor[(Drawings, Edge), Int32](read(10))
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
    *
    * @param afterSteps The batches already taken, which are passed over without being read: a run
    *                   that carries on from a checkpoint reads on where it left off.
    */
  def batches[S: Label](batch: AxisExtent[S], afterSteps: Int): Iterator[Batch[S, W, H, RecordBatch[S, Node, Edge]]] =
    require(batch.size <= numSamples, s"a batch of ${batch.size} exceeds the $numSamples drawings of the split")
    val starts = 0 to numSamples - batch.size by batch.size
    Iterator.continually(starts).flatten.drop(afterSteps).map(from => Batch(stored(batch, from), recordsIn(batch, from)))

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
      records.construction.slice(drawing),
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
      construction = liftPyTensor[(S, Node), Int32](cut(1)),
      startX = liftPyTensor[(S, Node), Float32](cut(2)),
      startY = liftPyTensor[(S, Node), Float32](cut(3)),
      endX = liftPyTensor[(S, Node), Float32](cut(4)),
      endY = liftPyTensor[(S, Node), Float32](cut(5)),
      midX = liftPyTensor[(S, Node), Float32](cut(6)),
      midY = liftPyTensor[(S, Node), Float32](cut(7)),
      edgeClass = liftPyTensor[(S, Edge), Int32](cut(8)),
      subject = liftPyTensor[(S, Edge), Int32](cut(9)),
      obj = liftPyTensor[(S, Edge), Int32](cut(10))
    )
