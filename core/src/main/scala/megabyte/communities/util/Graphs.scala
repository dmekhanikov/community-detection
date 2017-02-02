package megabyte.communities.util

import edu.uci.ics.jung.graph.util.EdgeType
import edu.uci.ics.jung.graph.{Graph, UndirectedSparseGraph}
import megabyte.communities.entities.Edge
import megabyte.communities.util.DoubleMatrixOps._
import org.jblas.DoubleMatrix
import org.jblas.DoubleMatrix._

import scala.collection.JavaConversions._

object Graphs {

  def laplacian(adj: DoubleMatrix): DoubleMatrix = {
    degreeMatrix(adj) - adj
  }

  def symLaplacian[V](adj: DoubleMatrix): DoubleMatrix = {
    val normDeg: DoubleMatrix = degreeMatrix(adj).sqrtDiagI().invDiagI()
    eye(adj.columns) - ((normDeg \* adj) *\= normDeg)
  }

  def adjacencyMatrix(graph: Graph[Int, Edge]): DoubleMatrix = {
    val n = graph.getVertices.max + 1
    val adj = zeros(n, n)
    for (e <- graph.getEdges) {
      val endpoints = graph.getEndpoints(e)
      val from = endpoints.getFirst
      val to = endpoints.getSecond
      adj.put(from, to, 1)
      if (graph.getEdgeType(e) == EdgeType.UNDIRECTED) {
        adj.put(to, from, 1)
      }
    }
    adj
  }

  def symAdjacencyMatrix(graph: Graph[Int, Edge], n: Int): DoubleMatrix = {
    val adj = zeros(n, n)
    for (e <- graph.getEdges) {
      val endpoints = graph.getEndpoints(e)
      val from = endpoints.getFirst
      val to = endpoints.getSecond
      val weight = adj.get(from, to) + 1
      adj.put(from, to, weight)
      adj.put(to, from, weight)
    }
    adj
  }

  def makeAdjBinary(adj: DoubleMatrix): DoubleMatrix = {
    for (i <- 0 until adj.rows; j <- 0 until adj.columns) {
      if (adj.get(i, j) != 0) {
        adj.put(i, j, 1)
      }
    }
    adj
  }

  def degreeMatrix(adj: DoubleMatrix): DoubleMatrix = {
    val degMatrix = zeros(adj.rows, adj.columns)
    val deg = degrees(adj)
    deg.zipWithIndex.foreach { case (d: Double, i: Int) => degMatrix.put(i, i, d) }
    degMatrix
  }

  def degrees(adj: DoubleMatrix): Seq[Double] = {
    (0 until adj.rows).map(adj.getRow(_).sum)
  }

  def numerateNodes[V](graph: Graph[V, Edge]): Seq[V] = {
    graph.getVertices.toSeq
  }

  def applyNumeration[V](graph: Graph[V, Edge], numeration: Seq[V]): Graph[Int, Edge] = {
    val inverseNumeration = Map[V, Int](numeration.zipWithIndex: _*)
    val newGraph = new UndirectedSparseGraph[Int, Edge]
    for (v <- numeration) {
      if (graph.containsVertex(v)) {
        graph.getIncidentEdges(v)
          .filter(e => !newGraph.containsEdge(e))
          .foreach { e =>
            if (!newGraph.containsEdge(e)) {
              val Seq(src, dst) = graph.getEndpoints(e).map(inverseNumeration)
              newGraph.addEdge(e, src, dst)
            }
          }
      }
    }
    newGraph
  }
}
