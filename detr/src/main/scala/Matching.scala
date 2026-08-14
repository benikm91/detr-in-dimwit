import dimwit.*
import dimwit.Conversions.given

import scala.language.implicitConversions

object Matching:

  /** Assigns every row a distinct column, repeatedly taking the cheapest remaining pair.
    *
    * This is greedy rather than optimal, which buys a fixed number of steps of plain tensor
    * operations: unlike an augmenting path algorithm it has no data dependent control flow,
    * so it traces, jits and vmaps like the rest of the model.
    */
  def greedy[Row: Label, Column: Label, V: IsFloating](cost: Tensor2[Row, Column, V]): Tensor1[Row, Int32] =
    val rows = cost.shape.extent(Axis[Row])
    val columns = cost.shape.extent(Axis[Column])
    val rowIndices = indices(rows)
    val columnIndices = indices(columns)
    // More than the whole spread of the matrix, so a taken pair always loses to an open one.
    val taken = (cost.max - cost.min + 1f) * (rows.size + 1).toFloat

    (0 until rows.size)
      .foldLeft((cost, Tensor(Shape1(rows), VType[Int32]).fill(0))):
        case ((remaining, assignment), _) =>
          val cheapestColumn = remaining.argmin(Axis[Column])
          val row = remaining.min(Axis[Column]).argmin
          val column = cheapestColumn.slice(Axis[Row].at(row))
          val isRow = rowIndices.elementEquals(row.broadcastTo(Shape1(rows)))
          val isColumn = columnIndices.elementEquals(column.broadcastTo(Shape1(columns)))
          val used = maximum(
            isRow.asFloat(VType[V]).broadcastTo(cost.shape),
            isColumn.asFloat(VType[V]).broadcastTo(cost.shape)
          )
          (
            remaining + used *! taken,
            where(isRow, column.broadcastTo(Shape1(rows)), assignment)
          )
      ._2

  private def indices[L: Label](extent: AxisExtent[L]): Tensor1[L, Int32] =
    Tensor1(extent.axis, VType[Int32]).fromArray(Array.range(0, extent.size))
