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
  .aggregate(dataset, detr, egtr)
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

lazy val modelSettings = Seq(
  libraryDependencies ++= Seq(
    dimwit,
    scalapy,
    munit,
    "ch.contrafactus" %% "plotwit-core" % "0.1.0-SNAPSHOT" changing (),
    "ch.contrafactus" %% "deepwit-core" % "0.1.0-SNAPSHOT" changing ()
  ),
  javaOptions ++= Seq(
    // "-XX:G1PeriodicGCInterval=1000"
    "-XX:+UseZGC",
    "-XX:ZCollectionInterval=1" // Forces a GC cycle every 1 second, regardless of heap usage
  )
)

lazy val detr = project
  .in(file("detr"))
  .dependsOn(dataset)
  .settings(name := "detr")
  .settings(modelSettings)

// EGTR builds its scene graph on the detector, so it depends on detr — never the other way round.
lazy val egtr = project
  .in(file("egtr"))
  .dependsOn(detr)
  .settings(name := "egtr")
  .settings(modelSettings)
