package megabyte.communities.util

import java.io.{BufferedReader, File, FileReader}

import com.typesafe.scalalogging.Logger
import edu.uci.ics.jung.graph.{DirectedSparseGraph, Graph, UndirectedSparseGraph}
import edu.uci.ics.jung.io.graphml._
import megabyte.communities.entities.Edge

import scala.language.implicitConversions

private class GraphFactory

object GraphFactory {

  private val logger = Logger[GraphFactory]

  val WEIGHT_PROP = "weight"

  implicit def fun2Guava[A, R](fn: (A) => R): com.google.common.base.Function[A, R] =
    new com.google.common.base.Function[A, R] {
      override def apply(arg: A): R = fn(arg)
    }

  private val graphTransformer = (metadata: GraphMetadata) => {
    metadata.getEdgeDefault match {
      case GraphMetadata.EdgeDefault.UNDIRECTED =>
        new UndirectedSparseGraph[String, Edge]()
      case GraphMetadata.EdgeDefault.DIRECTED =>
        new DirectedSparseGraph[String, Edge]()
      case _ => throw new IllegalArgumentException("Unsupported graph type")
    }
  }

  private val vertexTransformer = (metadata: NodeMetadata) => metadata.getId

  private val edgeTransformer = (metadata: EdgeMetadata) => {
    Option(metadata.getProperty(WEIGHT_PROP)) match {
      case Some(weight) => new Edge(weight.toDouble)
      case None => new Edge()
    }
  }

  private val hyperEdgeTransformer = (metadata: HyperEdgeMetadata) =>
    throw new IllegalArgumentException("Hyper-edges are not supported")

  def readGraph(file: File): Graph[String, Edge] = {
    logger.info(s"reading graph from file $file")
    val fileReader = new BufferedReader(new FileReader(file))
    val graphReader = new GraphMLReader2(
      fileReader,
      graphTransformer,
      vertexTransformer,
      edgeTransformer,
      hyperEdgeTransformer)
    val graph = graphReader.readGraph()
    logger.info(s"graph is successfully read from file $file. Nodes: ${graph.getVertexCount}; edges: ${graph.getEdgeCount()}")
    graph
  }
}
