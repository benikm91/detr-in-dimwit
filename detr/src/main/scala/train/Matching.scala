package detr.train

import detr.*
import detr.model.*
import detr.eval.*
import detr.config.*
import dimwit.*
import dimwit.python.PyBridge.liftPyTensor1
import dimwit.python.PyBridge.toPyTensor
import me.shadaj.scalapy.py

import scala.language.implicitConversions

object Matching:

  /** Assigns every row a distinct column of a square cost so that the assignment costs the least
    * in total.
    *
    * The Hungarian algorithm of [[https://optax.readthedocs.io/en/latest/api/assignment.html Optax]]
    * is written in JAX, so it traces, jits and vmaps like the rest of the model.
    */
  def optimal[Row: Label, Column: Label, V: IsFloating](cost: Tensor2[Row, Column, V]): Tensor1[Row, Int32] =
    require(cost.shape(Axis[Row]) == cost.shape(Axis[Column]), s"a cost of ${cost.shape} is not square")
    // Optax answers column by column: the row each column goes to.
    val rowOfColumn = liftPyTensor1(Axis[Column], VType[Int32])(
      py.module("optax.assignment").hungarian_algorithm(toPyTensor(cost)).bracketAccess(0)
    )
    rowOfColumn.argsort(Axis[Column]).relabelTo(Axis[Row])

  /** What each row pays for the column [[optimal]] assigned it. */
  def costOf[Row: Label, Column: Label, V: IsFloating](cost: Tensor2[Row, Column, V], assignment: Tensor1[Row, Int32]): Tensor1[Row, V] =
    val isAssigned = indices(cost.shape.extent(Axis[Column]))
      .broadcastTo(cost.shape)
      .elementEquals(assignment.broadcastTo(cost.shape))
    where(isAssigned, cost, Tensor.like(cost).fill(0f)).sum(Axis[Column])

  private def indices[L: Label](extent: AxisExtent[L]): Tensor1[L, Int32] =
    Tensor1(extent.axis, VType[Int32]).fromArray(Array.range(0, extent.size))
