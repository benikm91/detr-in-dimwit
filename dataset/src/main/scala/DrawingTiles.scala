package dataset

import dimwit.*
import dimwit.python.PyBridge.toPyTensor

import java.nio.file.Path

/** Many drawings in one picture, written as a PNG by `drawing_tiles.py`. */
object DrawingTiles:

  /** @param eachDrawing the tiles of one drawing, laid side by side; every drawing has as many. */
  def write[W: Label, H: Label, C: Label](eachDrawing: Seq[Seq[Tensor3[W, H, C, UInt8]]], rows: Int, across: Int, picture: Path): Unit =
    module.write(toPyTensor(stack(eachDrawing.flatten, Axis[Tile])), picture.toString, rows, across, eachDrawing.head.size)

  private trait Tile derives Label

  private lazy val module = PythonModules("drawing_tiles")
