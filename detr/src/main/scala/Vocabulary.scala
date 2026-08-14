import dataset.Detection
import dimwit.*

trait Width derives Label
trait Height derives Label
trait Channel derives Label

/** Axis of the detected objects, i.e. of DETR's object queries. */
trait BoundingBox derives Label

/** Axis over the `ObjectClass` values a box is classified into. */
trait ObjectClasses derives Label

type ObjectDetection = Detection[BoundingBox]
