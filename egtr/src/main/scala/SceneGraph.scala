package egtr

import dataset.RelationClasses
import detr.*
import dimwit.*

/** The objects of a drawing and the relations between them.
  *
  * `relations` is the grid over ordered pairs of the [[Detection]]'s slots: entry `(i, j, k)`
  * is how strongly the object in slot `i` carries the `k`-th [[dataset.RelationClass]] towards
  * the object in slot `j`. A target holds ones and zeros, a prediction a score in between —
  * the same type either way, as with [[Detection]] itself, so that targets and predictions are
  * scored and drawn by the same code.
  */
case class SceneGraph[V](
    objects: ObjectDetection[V],
    relations: Tensor3[BoundingBox, RelatedBox, RelationClasses, V]
)
