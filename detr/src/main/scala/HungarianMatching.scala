import dimwit.*

/** Solves the linear assignment problem: assigns every row a distinct column such that the
  * total cost is minimal.
  *
  * The assignment is a discrete decision, so this runs on the host and cannot be traced by
  * `jit`.
  */
object HungarianMatching:

  def apply[Row: Label, Column: Label, V: IsFloating](cost: Tensor2[Row, Column, V]): Tensor1[Row, Int32] =
    val rows = cost.shape(Axis[Row])
    val columns = cost.shape(Axis[Column])
    require(rows <= columns, s"cannot assign $rows rows to $columns columns")
    Tensor1(Axis[Row], VType[Int32]).fromArray(assign(cost.asFloat32.toArray, rows, columns))

  /** Shortest augmenting path algorithm, with potentials `u`, `v` keeping the reduced costs
    * non-negative. Column 0 is the virtual start of a path, so rows and columns are indexed
    * from 1 here.
    */
  private def assign(cost: Array[Array[Float]], rows: Int, columns: Int): Array[Int] =
    val u = Array.fill(rows + 1)(0.0)
    val v = Array.fill(columns + 1)(0.0)
    val rowOf = Array.fill(columns + 1)(0)
    val previous = Array.fill(columns + 1)(0)

    for row <- 1 to rows do
      rowOf(0) = row
      var column = 0
      val minimal = Array.fill(columns + 1)(Double.PositiveInfinity)
      val visited = Array.fill(columns + 1)(false)

      var free = false
      while !free do
        visited(column) = true
        val current = rowOf(column)
        var delta = Double.PositiveInfinity
        var next = 0
        for candidate <- 1 to columns do
          if !visited(candidate) then
            val reduced = cost(current - 1)(candidate - 1) - u(current) - v(candidate)
            if reduced < minimal(candidate) then
              minimal(candidate) = reduced
              previous(candidate) = column
            if minimal(candidate) < delta then
              delta = minimal(candidate)
              next = candidate
        for candidate <- 0 to columns do
          if visited(candidate) then
            u(rowOf(candidate)) += delta
            v(candidate) -= delta
          else minimal(candidate) -= delta
        column = next
        free = rowOf(column) == 0

      while column != 0 do
        val source = previous(column)
        rowOf(column) = rowOf(source)
        column = source

    val assignment = Array.fill(rows)(0)
    for column <- 1 to columns do
      if rowOf(column) != 0 then assignment(rowOf(column) - 1) = column - 1
    assignment
