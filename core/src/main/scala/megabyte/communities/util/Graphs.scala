package megabyte.communities.util

import edu.uci.ics.jung.graph.util.EdgeType
import edu.uci.ics.jung.graph.{Graph, UndirectedSparseGraph}
import megabyte.communities.entities.Edge
import megabyte.communities.util.Matrices._
import org.jblas.DoubleMatrix
import org.jblas.DoubleMatrix._

import scala.collection.JavaConversions._

object Graphs {

  def laplacian(adj: DoubleMatrix): DoubleMatrix = {
    val deg = degreeMatrix(adj)
    deg.subi(adj)
  }

  def symLaplacian[V](adj: DoubleMatrix): DoubleMatrix = {
    val normDeg: DoubleMatrix = degreeMatrix(adj).sqrtDiagI().invDiagI()
    val id = eye(adj.columns)
    id.subi(normDeg.mulDiag(adj).mulByDiagI(normDeg))
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
    val deg = zeros(adj.rows, adj.columns)
    for (i <- 0 until adj.columns) {
      deg.put(i, i, adj.getColumn(i).sum())
    }
    deg
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
