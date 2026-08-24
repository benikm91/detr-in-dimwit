import dataset.NodeClass
import dataset.NodeClasses
import dataset.Record
import dimwit.*
import dimwit.Conversions.given

import scala.language.implicitConversions

/** Axis of the decoder's sequence: a node embedding per [[Node]], then a prediction embedding
  * per [[Node]].
  */
trait DecoderSlot derives Label

/** The decoder sequence of remaining-node prediction.
  *
  * An embedding in a decoder has two jobs — become what its position predicts, and keep carrying
  * what its position holds for the others to read. Next-token prediction can do both at once;
  * remaining-node prediction cannot, since the later slots have to know what is taken already in
  * order to answer with something else. So every slot gets both a *node embedding* carrying the
  * taken node and a *prediction embedding* becoming one of the remaining ones.
  *
  * The paper interleaves the two. Here they are appended, which the attention cannot tell apart
  * from any other layout and which leaves slot `i` meaning the same thing in both halves.
  */
object DecoderSequence:

  def join[Embedding: Label, V](
      taken: Tensor2[Node, Embedding, V],
      remaining: Tensor2[Node, Embedding, V]
  ): Tensor2[DecoderSlot, Embedding, V] =
    concatenate(taken, remaining, Axis[Node]).relabel(Axis[Node] -> Axis[DecoderSlot])

  /** The half that carried the taken nodes, i.e. what the pass-through loss reads. */
  def taken[Embedding: Label, V](sequence: Tensor2[DecoderSlot, Embedding, V]): Tensor2[Node, Embedding, V] =
    half(sequence, 0)

  /** The half that became the remaining nodes, i.e. what a transcription is read off. */
  def remaining[Embedding: Label, V](sequence: Tensor2[DecoderSlot, Embedding, V]): Tensor2[Node, Embedding, V] =
    half(sequence, 1)

  /** Every embedding attends to itself and, beyond that, only to taken nodes: a node embedding to
    * those up to its own slot, a prediction embedding to those strictly before it, so that what it
    * may answer with is exactly what is left over. Nothing reads a prediction embedding, which
    * carries a guess rather than a node.
    */
  def mask(record: Record[Node]): Tensor2[DecoderSlot, DecoderSlot, Bool] =
    val slots = record.nodeClass.shape.extent(Axis[Node])
    val sequence = Axis[DecoderSlot] -> 2 * slots.size
    val pairs = Shape2(sequence, Axis[Prime[DecoderSlot]] -> sequence.size)

    def halves[V](node: Tensor1[Node, V], prediction: Tensor1[Node, V]): Tensor1[DecoderSlot, V] =
      concatenate(node, prediction, Axis[Node]).relabelTo(Axis[DecoderSlot])
    def attended[V](of: Tensor1[DecoderSlot, V]): Tensor2[DecoderSlot, Prime[DecoderSlot], V] =
      of.relabelTo(Axis[Prime[DecoderSlot]]).broadcastTo(pairs)

    val slot = Tensor1(Axis[Node], VType[Int32]).fromArray(Array.range(0, slots.size))
    val holdsNode = NodeClass.indicator(VType[Float32])(_ != NodeClass.NoNode).take(Axis[NodeClasses])(record.nodeClass) > Tensor1(slots).fill(0f)
    val carries = halves(holdsNode, Tensor1(slots).fill(false))
    val reach = halves(slot +! 1, slot).broadcastTo(pairs)
    val position = Tensor1(sequence, VType[Int32]).fromArray(Array.range(0, sequence.size))
    val itself = Tensor2.eye(sequence, VType[Bool])

    where(attended(carries), attended(position) < reach, itself)
      .relabel(Axis[Prime[DecoderSlot]] -> Axis[DecoderSlot])

  private def half[Embedding: Label, V](sequence: Tensor2[DecoderSlot, Embedding, V], index: Int): Tensor2[Node, Embedding, V] =
    val slots = sequence.shape(Axis[DecoderSlot]) / 2
    sequence
      .slice(Axis[DecoderSlot].at(index * slots until (index + 1) * slots))
      .relabel(Axis[DecoderSlot] -> Axis[Node])
