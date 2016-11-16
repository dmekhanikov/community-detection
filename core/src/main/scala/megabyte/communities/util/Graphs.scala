package megabyte.communities.util

import edu.uci.ics.jung.graph.{Graph, UndirectedSparseGraph}
import megabyte.communities.entities.Edge
import org.jblas.DoubleMatrix
import org.jblas.DoubleMatrix._
import org.jblas.MatrixFunctions._
import org.jblas.Solve.pinv

import scala.collection.JavaConversions._

object Graphs {

  def laplacian(adj: DoubleMatrix): DoubleMatrix = {
    val deg = degreeMatrix(adj)
    deg.subi(adj)
  }

  def symLaplacian[V](adj: DoubleMatrix): DoubleMatrix = {
    val normDeg: DoubleMatrix = pinv(sqrti(degreeMatrix(adj)))
    val id = eye(adj.columns)
    id.subi(normDeg.mmul(adj).mmuli(normDeg))
  }

  def adjacencyMatrix(graph: Graph[Int, Edge]): DoubleMatrix = {
    val adj = zeros(graph.getVertexCount, graph.getVertexCount)
    for (e <- graph.getEdges) {
      val endpoints = graph.getEndpoints(e)
      val from = endpoints.getFirst
      val to = endpoints.getSecond
      adj.put(from, to, 1)
      adj.put(to, from, 1)
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
    for ((v, i) <- numeration.zipWithIndex) {
      graph.getIncidentEdges(v)
        .filter(e => !newGraph.containsEdge(e))
        .foreach { e =>
          if (!newGraph.containsEdge(e)) {
            val Seq(src, dst) = graph.getEndpoints(e).map(inverseNumeration)
            newGraph.addEdge(e, src, dst)
          }
        }
    }
    newGraph
  }
}
