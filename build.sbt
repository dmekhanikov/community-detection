name := "community-detection"

version := "1.0"

scalaVersion := "2.11.8"

libraryDependencies ++= Seq(
  "net.sf.jung" % "jung-graph-impl" % "2.1",
  "net.sf.jung" % "jung-io" % "2.1",
  "org.jblas" % "jblas" % "1.2.4",
  "nz.ac.waikato.cms.weka" % "weka-stable" % "3.8.0",

  "org.slf4j" % "slf4j-api" % "1.7.21",
  "org.slf4j" % "slf4j-simple" % "1.7.21"
)

resolvers += "Sonatype Releases" at "https://oss.sonatype.org/content/repositories/releases/"
