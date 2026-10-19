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

lazy val dimwit = "ch.contrafactus" %% "dimwit-core" % "0.3-SNAPSHOT" changing ()
lazy val dimwitSharding = "ch.contrafactus" %% "dimwit-sharding" % "0.1.0"
lazy val deepwit = "ch.contrafactus" %% "deepwit-core" % "0.3-SNAPSHOT" changing ()
// plotwit has no release yet, and its snapshot depends on a DimWit snapshot: it uses ours instead.
lazy val plotwit = ("ch.contrafactus" %% "plotwit-core" % "0.2-SNAPSHOT" changing ()).exclude("ch.contrafactus", "dimwit-core_3")
lazy val scalapy = "dev.scalapy" %% "scalapy-core" % "0.5.3"
lazy val munit = "org.scalameta" %% "munit" % "1.0.0" % Test

lazy val root = project
  .in(file("."))
  .aggregate(dataset, syntheticFloorplansDataset, documentEncoder, detr, detrBucket, d2s, d2g, d2seq, d2tok)
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

lazy val syntheticFloorplansDataset = project
  .in(file("syntheticFloorplansDataset"))
  .dependsOn(dataset) // The batches and samples every model reads, filled from another corpus.
  .settings(
    name := "syntheticFloorplansDataset",
    libraryDependencies ++= Seq(dimwit, scalapy, munit)
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
  .dependsOn(dataset, syntheticFloorplansDataset, documentEncoder)
  .settings(name := "detr")
  .settings(modelSettings)

lazy val detrBucket = project
  .in(file("detr-bucket"))
  .dependsOn(detr) // DETR's decoder and matching, with queries split by class.
  .settings(name := "detr-bucket")
  .settings(modelSettings)

lazy val d2s = project
  .in(file("d2s"))
  .dependsOn(dataset, syntheticFloorplansDataset, documentEncoder)
  .settings(name := "d2s")
  .settings(modelSettings)

lazy val d2g = project
  .in(file("d2g"))
  .dependsOn(d2s) // D2G is a D2S with relationships on top, so it depends on d2s.
  .settings(name := "d2g")
  .settings(modelSettings)

lazy val d2seq = project
  .in(file("d2seq"))
  .dependsOn(dataset, documentEncoder) // A sequence model to set beside d2s, sharing none of its code.
  .settings(name := "d2seq")
  .settings(modelSettings)

lazy val d2tok = project
  .in(file("d2tok"))
  .dependsOn(dataset, syntheticFloorplansDataset, documentEncoder) // A language model of records to set beside d2s, sharing none of its code.
  .settings(name := "d2tok")
  .settings(modelSettings)
