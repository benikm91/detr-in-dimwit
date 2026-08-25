package dataset

import dimwit.*
import dimwit.jax.Jax
import dimwit.python.PyBridge.liftPyTensor
import dimwit.python.PyBridge.toPyTensor
import me.shadaj.scalapy.py
import me.shadaj.scalapy.py.SeqConverters

/** Draws a record as the picture it stands for, over the drawing it was read from.
  *
  * A record has no drawing of its own, so a transcription can only be looked at by drawing it:
  * see `record_drawing.py`, which is where the drawing happens.
  */
object RecordDrawing:

  def apply[W: Label, H: Label, C: Label](record: RecordGraph, over: Tensor2[W, H, UInt8], colour: Axis[C]): Tensor3[W, H, C, UInt8] =
    val placed = record.placed(record.nodes.size, record.edges.size)
    liftPyTensor[(W, H, C), UInt8](
      Jax.jnp.asarray(
        module.render(
          toPyTensor(over),
          placed.nodeClass.toSeq.toPythonCopy,
          placed.startX.toSeq.toPythonCopy,
          placed.startY.toSeq.toPythonCopy,
          placed.endX.toSeq.toPythonCopy,
          placed.endY.toSeq.toPythonCopy,
          placed.edgeClass.toSeq.toPythonCopy,
          placed.subject.toSeq.toPythonCopy,
          placed.obj.toSeq.toPythonCopy,
          NodeClass.Line.id,
          NodeClass.Annotation.id,
          EdgeClass.Connected.id,
          EdgeClass.Annotates.id
        )
      )
    )

  private lazy val module: py.Dynamic = PythonModules("record_drawing")
