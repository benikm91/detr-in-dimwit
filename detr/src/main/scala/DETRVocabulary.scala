package detr

import detr.model.*
import detr.train.*
import detr.eval.*
import detr.config.*
import dataset.Detection
import dimwit.*

export documentEncoder.{Width, Height, Channel, Patch}

/** Axis of the detected objects, i.e. of DETR's object queries. */
trait BoundingBox derives Label

/** Axis over the `ObjectClass` values a box is classified into. */
trait ObjectClasses derives Label

/** Axis of the relationships a record holds between its nodes, which a detector does not predict
  * but the records it is trained on still carry.
  */
trait Relationship derives Label

type ObjectDetection[V] = Detection[BoundingBox, V]
