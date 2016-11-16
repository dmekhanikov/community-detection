import sbt.Keys._

name := "community-detection"

version in ThisBuild := "1.0"

scalaVersion in ThisBuild := "2.11.8"

lazy val root = project.in(file(".")).aggregate(core, examples)

lazy val core = project

lazy val examples = project.dependsOn(core)

libraryDependencies in ThisBuild ++= Seq(
  "net.sf.jung" % "jung-graph-impl" % "2.1",
  "net.sf.jung" % "jung-io" % "2.1",
  "org.jblas" % "jblas" % "1.2.4",
  "nz.ac.waikato.cms.weka" % "XMeans" % "1.0.4",

  "org.slf4j" % "slf4j-api" % "1.7.21",
  "org.slf4j" % "slf4j-simple" % "1.7.21"
)

resolvers in ThisBuild += "Sonatype Releases" at "https://oss.sonatype.org/content/repositories/releases/"
