package megabyte.communities.util

import edu.uci.ics.jung.graph.Graph
import megabyte.communities.entities.Edge
import org.jblas.DoubleMatrix
import DoubleMatrixOps._

import scala.collection.JavaConversions._
import scala.collection.Map

object Measures {

  def euclidNorm(v: Seq[Double]): Double = {
    val sqrSum = v.map { x => x * x }.sum
    math.sqrt(sqrSum)
  }

  def gaussianSim(p1: DoubleMatrix, p2: DoubleMatrix, sigma: Double): Double = {
    val d = p1.distTo(p2)
    Math.exp(-math.pow(d, 2) / 2 / math.pow(sigma, 2))
  }

  def modularity[V](graph: Graph[V, Edge], clustering: Map[V, Int]): Double = {
    val m = graph.getEdges.map(e => e.weight).sum
    val deg = graph.getVertices.map(v => v -> graph.getOutEdges(v).map(e => e.weight).sum).toMap
    var sum = 0.0
    for (i <- graph.getVertices; j <- graph.getVertices) {
      if (clustering(i) == clustering(j)) {
        Option(graph.findEdge(i, j)) match {
          case Some(edge) =>
            sum += edge.weight - deg(i) * deg(j) / 2 / m
          case None =>
        }
      }
    }
    sum / 2 / m
  }

  def modularity(adj: DoubleMatrix, clustering: Seq[Int]): Double = {
    val n = adj.rows
    val k = clustering.max + 1
    val s = DoubleMatrix.zeros(n, k)
    val deg = Graphs.degrees(adj)
    val m = deg.sum / 2
    for (vertex <- 0 until n) {
      val cluster = clustering(vertex)
      s.put(vertex, cluster, 1)
    }
    val b = DoubleMatrix.zeros(n, n)
    for (i <- 0 until n; j <- 0 until n) {
      val v = adj.get(i, j) - deg(i) * deg(j) / 2 / m
      b.put(i, j, v)
    }
    (s.transpose * b * s).diag.sum / 2 / m
  }
}
