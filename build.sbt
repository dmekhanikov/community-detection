import sbt.Keys._

name := "community-detection"

version in ThisBuild := "1.0"

scalaVersion in ThisBuild := "2.11.8"

lazy val root = project.in(file(".")).aggregate(core, examples, experiments)

lazy val core = project

lazy val examples = project.dependsOn(core)

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
  "net.sf.jung" % "jung-graph-impl" % versions.jung,
  "net.sf.jung" % "jung-io" % versions.jung,
  "org.jblas" % "jblas" % "1.2.4",
  "nz.ac.waikato.cms.weka" % "XMeans" % "1.0.4",

  "org.slf4j" % "slf4j-api" % versions.slf4j,
  "org.slf4j" % "slf4j-simple" % versions.slf4j,
  "com.typesafe.scala-logging" %% "scala-logging" % "3.5.0"
)

resolvers in ThisBuild += "Sonatype Releases" at "https://oss.sonatype.org/content/repositories/releases/"
