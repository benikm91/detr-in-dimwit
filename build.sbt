import scala.concurrent.duration.DurationInt
import lmcoursier.definitions.CachePolicy

ThisBuild / version := "0.2-SNAPSHOT"
ThisBuild / scalaVersion := "3.9.0"
ThisBuild / fork := true
ThisBuild / envVars ++= sys.env
ThisBuild / resolvers += Resolver.defaultLocal
ThisBuild / csrConfiguration := csrConfiguration.value
  .withTtl(Some(0.seconds))
  .withCachePolicies(Vector(CachePolicy.LocalOnly))

lazy val dimwit = "ch.contrafactus" %% "dimwit-core" % "0.2.0"
lazy val dimwitSharding = "ch.contrafactus" %% "dimwit-sharding" % "0.1.0"
lazy val deepwit = "ch.contrafactus" %% "deepwit-core" % "0.2.1"
// plotwit has no release yet, and its snapshot depends on a DimWit snapshot: it uses ours instead.
lazy val plotwit = ("ch.contrafactus" %% "plotwit-core" % "0.2-SNAPSHOT" changing ()).exclude("ch.contrafactus", "dimwit-core_3")
lazy val scalapy = "dev.scalapy" %% "scalapy-core" % "0.5.3"
lazy val munit = "org.scalameta" %% "munit" % "1.0.0" % Test

lazy val root = project
  .in(file("."))
  .aggregate(dataset, documentEncoder, detr, d2g)
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
      plotwit
    )
  )

lazy val documentEncoder = project
  .in(file("documentEncoder"))
  .settings(
    name := "documentEncoder",
    libraryDependencies ++= Seq(dimwit, deepwit, munit)
  )

lazy val modelSettings = Seq(
  libraryDependencies ++= Seq(
    dimwit,
    dimwitSharding,
    deepwit,
    scalapy,
    munit,
    plotwit
  ),
  javaOptions ++= Seq(
    // "-XX:G1PeriodicGCInterval=1000"
    "-XX:+UseZGC",
    "-XX:ZCollectionInterval=1" // Forces a GC cycle every 1 second, regardless of heap usage
  )
)

lazy val detr = project
  .in(file("detr"))
  .dependsOn(dataset, documentEncoder)
  .settings(name := "detr")
  .settings(modelSettings)

lazy val d2g = project
  .in(file("d2g"))
  .dependsOn(dataset, documentEncoder)
  .settings(name := "d2g")
  .settings(modelSettings)
