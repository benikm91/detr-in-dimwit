package d2s.eval

import dimwit.*

/** The log probability of the value a position's scores are highest for. */
def chosen[Slot: Label, L: Label](logits: Tensor2[Slot, L, Float32]): Tensor1[Slot, Float32] =
  logits.logSoftmax(Axis[L]).max(Axis[L])
