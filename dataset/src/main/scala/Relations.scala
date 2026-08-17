package dataset

import dimwit.*

/** Axis over the [[RelationClass]] values an edge is classified into. */
trait RelationClasses derives Label

/** The edges of a drawing program, between the objects a [[Detection]] holds.
  *
  * Unlike [[ObjectClass]] there is no "no relation" class: a pair of objects carries each
  * class or it does not, independently of the others, so an unrelated pair is simply zero
  * everywhere. That is what makes the relation head a set of binary decisions rather than a
  * classification, and it is why two objects may be related in more than one way at once.
  */
enum RelationClass(val id: Int):

  /** Two [[ObjectClass.PartLine]]s that meet in a corner of the outline. */
  case Connected extends RelationClass(0)

  /** An [[ObjectClass.Text]] and the [[ObjectClass.PartLine]] whose length it annotates. */
  case Annotates extends RelationClass(1)

object RelationClass:

  def fromId(id: Int): RelationClass =
    values.find(_.id == id).getOrElse(throw IllegalArgumentException(s"unknown relation class id: $id"))

  /** How a drawing program spells a relation out.
    *
    * A drawing program names its elements by an id, counted over the actions of
    * [[idActionTypes]], which a relationship action then refers to in its discrete
    * parameters.
    *
    * @param actionType The action carrying the relation.
    * @param kind       `between` — the action's two trailing discrete parameters are the ids
    *                   of the two elements it relates; `from_self` — the action is itself an
    *                   object and its trailing discrete parameter is the id of the element it
    *                   refers to.
    * @param symmetric  Whether the relation holds in both directions. Which way round a
    *                   relationship action names its two elements carries no meaning, so a
    *                   symmetric relation is stored both ways and predicted both ways.
    */
  private[dataset] case class Action(actionType: String, kind: String, symmetric: Boolean)

  private[dataset] val actions: Map[RelationClass, Action] = Map(
    Connected -> Action("ConnectTwoElementsWithId", "between", symmetric = true),
    Annotates -> Action("AnnotationTextRefId", "from_self", symmetric = false)
  )

  /** The actions that spend an element id, i.e. the ones a relation can refer to. */
  private[dataset] val idActionTypes: Seq[String] = Seq("PartLineWithId")
