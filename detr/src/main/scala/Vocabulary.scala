import dataset.Detection
import dimwit.*

trait Width derives Label
trait Height derives Label
trait Channel derives Label

/** Axis of the detected objects, i.e. of DETR's object queries. */
trait BoundingBox derives Label

/** Axis over the `ObjectClass` values a box is classified into. */
trait ObjectClasses derives Label

/** Axis of the relationships a record holds between its nodes, which a detector does not predict
  * but the records it is trained on still carry.
  */
trait Relationship derives Label

type ObjectDetection[V] = Detection[BoundingBox, V]
