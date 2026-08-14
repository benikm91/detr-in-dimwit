import scala.concurrent.duration.DurationInt
import lmcoursier.definitions.CachePolicy

ThisBuild / version := "0.1.0-SNAPSHOT"
ThisBuild / scalaVersion := "3.8.1"
ThisBuild / fork := true
ThisBuild / envVars ++= sys.env
ThisBuild / resolvers += Resolver.defaultLocal
ThisBuild / csrConfiguration := csrConfiguration.value
  .withTtl(Some(0.seconds))
  .withCachePolicies(Vector(CachePolicy.LocalOnly))

lazy val dimwit = "ch.contrafactus" %% "dimwit-core" % "0.1.0-SNAPSHOT" changing ()
lazy val scalapy = "dev.scalapy" %% "scalapy-core" % "0.5.3"
lazy val munit = "org.scalameta" %% "munit" % "1.0.0" % Test

lazy val root = project
  .in(file("."))
  .aggregate(dataset, detr)
  .settings(
    name := "detr-root",
    publish / skip := true
  )

lazy val dataset = project
  .in(file("dataset"))
  .settings(
    name := "dataset",
    libraryDependencies ++= Seq(
      dimwit,
      scalapy,
      munit,
      "ch.contrafactus" %% "plotwit-core" % "0.1.0-SNAPSHOT" changing ()
    )
  )

lazy val detr = project
  .in(file("detr"))
  .dependsOn(dataset)
  .settings(
    name := "detr",
    libraryDependencies ++= Seq(
      dimwit,
      scalapy,
      munit,
      "ch.contrafactus" %% "plotwit-core" % "0.1.0-SNAPSHOT" changing (),
      "ch.contrafactus" %% "deepwit-core" % "0.1.0-SNAPSHOT" changing ()
    )
  )
