package egtr

import detr.*
import dimwit.*

/** The axis of the object of a relation.
  *
  * The subject is [[BoundingBox]], so a relation representation is one entry of a square grid
  * over the query slots: row `i`, column `j` is what query `i` as a subject makes of query `j`
  * as an object. The two are the same slots, which is why this is the primed axis rather than
  * one of its own.
  */
type RelatedBox = Prime[BoundingBox]

/** Axis of a relation representation, i.e. of the subject and object halves of a pair
  * concatenated.
  */
trait RelationSource derives Label

/** Axis of the hidden space of the relation and connectivity heads. */
trait RelationHidden derives Label

/** Axis of the single score the connectivity head gives a pair. */
trait Connectivity derives Label
