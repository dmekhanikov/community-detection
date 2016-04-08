package megabyte.communities

import java.io.{BufferedReader, File, FileReader}

import edu.uci.ics.jung.graph.{UndirectedGraph, UndirectedSparseGraph}
import edu.uci.ics.jung.io.graphml._

object GraphFactory {

  val WEIGHT_PROP = "weight"

  implicit def fun2Guava[A, R](fn: (A) => R): com.google.common.base.Function[A, R] =
    new com.google.common.base.Function[A, R] {
      override def apply(arg: A): R = fn(arg)
    }

  private val graphTransformer = (metadata: GraphMetadata) => {
    if (metadata.getEdgeDefault == GraphMetadata.EdgeDefault.UNDIRECTED) {
      new UndirectedSparseGraph[Long, Edge]()
    } else {
      throw new IllegalArgumentException("Directed graphs are not supported")
    }
  }

  private val vertexTransformer = (metadata: NodeMetadata) => metadata.getId.toLong

  private val edgeTransformer = (metadata: EdgeMetadata) => {
    if (metadata.isDirected) {
      throw new IllegalArgumentException("Directed edges are not supported")
    }
    val weight = metadata.getProperty(WEIGHT_PROP).toDouble
    new Edge(weight)
  }

  private val hyperEdgeTransformer = (metadata: HyperEdgeMetadata) =>
    throw new IllegalArgumentException("Hyper-edges are not supported")

  def readGraph(file: File): UndirectedGraph[Long, Edge] = {
    val fileReader = new BufferedReader(new FileReader(file))
    val graphReader = new GraphMLReader2(
      fileReader,
      graphTransformer,
      vertexTransformer,
      edgeTransformer,
      hyperEdgeTransformer)
    graphReader.readGraph()
  }
}
