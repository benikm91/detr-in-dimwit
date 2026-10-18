package detr.train

import dimwit.*

import scala.language.implicitConversions

/** The cross entropy of every query's scores against the value every node holds. */
def costOfValue[Query: Label, Node: Label, L: Label, V: IsFloating](logits: Tensor2[Query, L, V], values: Tensor1[Node, Int32]): Tensor2[Query, Node, V] =
  logNormalizer(logits) -! logits.slice(Axis[L].at(values))

/** The cross entropy of every query's scores against one class. */
def costOfClass[Query: Label, Classes: Label, V: IsFloating](logits: Tensor2[Query, Classes, V], id: Int): Tensor1[Query, V] =
  logNormalizer(logits) - logits.slice(Axis[Classes].at(id))

def logNormalizer[Query: Label, L: Label, V: IsFloating](logits: Tensor2[Query, L, V]): Tensor1[Query, V] =
  val peak = logits.max(Axis[L])
  peak + (logits -! peak).exp.sum(Axis[L]).log
