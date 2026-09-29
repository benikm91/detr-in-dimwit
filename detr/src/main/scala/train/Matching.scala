package detr.train

import detr.*
import dimwit.*
import dimwit.python.PyBridge.liftPyTensor1
import dimwit.python.PyBridge.toPyTensor
import me.shadaj.scalapy.py

import scala.language.implicitConversions

object Matching:

  /** For every column, the row it is assigned: every column a distinct row, so that the assignment
    * costs the least in total. The rows no column is assigned are left over.
    *
    * The Hungarian algorithm of [[https://optax.readthedocs.io/en/latest/api/assignment.html Optax]]
    * is written in JAX, so it traces, jits and vmaps like the rest of the model.
    */
  def optimal[Row: Label, Column: Label, V: IsFloating](cost: Tensor2[Row, Column, V]): Tensor1[Column, Int32] =
    require(cost.shape(Axis[Row]) >= cost.shape(Axis[Column]), s"a cost of ${cost.shape} leaves columns without a row")
    // Optax answers with the assigned pairs, a row and a column each, in no particular order.
    val pairs = py.module("optax.assignment").hungarian_algorithm(toPyTensor(cost))
    val columns = cost.shape.extent(Axis[Column])
    def read(at: Int) = liftPyTensor1(columns.axis, VType[Int32])(pairs.bracketAccess(at))
    val (row, column) = (read(0), read(1))
    // Sorted by their column, every column's row lands at that column.
    row.take(Axis[Column])(column.argsort(Axis[Column]))
