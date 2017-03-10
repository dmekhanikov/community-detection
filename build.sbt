import sbt.Keys._

name := "community-detection"

version in ThisBuild := "1.0"

scalaVersion in ThisBuild := "2.11.8"

lazy val root = project.in(file(".")).aggregate(core, visualization, experiments)

lazy val core = project

lazy val visualization = project.dependsOn(core)

lazy val experiments = project.dependsOn(core)

val versions = new {
  val jung = "2.1"
  val slf4j = "1.7.21"
}

libraryDependencies in experiments ++= Seq(
  "org.mongodb" % "mongo-java-driver" % "3.4.1",
  "org.apache.commons" % "commons-csv" % "1.4"
)

libraryDependencies in ThisBuild ++= Seq(
  "org.scalaz" % "scalaz-core_2.11" % "7.3.0-M8",

  "net.sf.jung" % "jung-graph-impl" % versions.jung,
  "net.sf.jung" % "jung-io" % versions.jung,
  "org.jblas" % "jblas" % "1.2.4",
  "nz.ac.waikato.cms.weka" % "XMeans" % "1.0.4",

  "org.slf4j" % "slf4j-api" % versions.slf4j,
  "org.slf4j" % "slf4j-simple" % versions.slf4j,
  "com.typesafe.scala-logging" %% "scala-logging" % "3.5.0",

  "org.scalactic" %% "scalactic" % "3.0.1",
  "org.scalatest" %% "scalatest" % "3.0.1" % "test"
)

scalacOptions += "-feature"

resolvers in ThisBuild += "Sonatype Releases" at "https://oss.sonatype.org/content/repositories/releases/"
