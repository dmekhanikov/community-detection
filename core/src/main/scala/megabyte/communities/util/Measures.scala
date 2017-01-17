package megabyte.communities.util

import edu.uci.ics.jung.graph.Graph
import megabyte.communities.entities.Edge
import org.jblas.DoubleMatrix

import scala.collection.JavaConversions._
import scala.collection.Map

object Measures {

  def euclidDist(p1: DoubleMatrix, p2: DoubleMatrix): Double = {
    euclidNorm(p1.sub(p2))
  }

  def euclidNorm(p: DoubleMatrix): Double = {
    val n = p.length
    val sqrSum = (0 until n)
      .map { i => math.pow(p.get(i), 2) }
      .sum
    math.sqrt(sqrSum)
  }

  def gaussianSim(p1: DoubleMatrix, p2: DoubleMatrix, sigma: Double): Double = {
    val d = euclidDist(p1, p2)
    Math.exp(-math.pow(d, 2) / 2 / math.pow(sigma, 2))
  }

  def modularity[V](graph: Graph[V, Edge], clustering: Map[V, Int]): Double = {
    val m = graph.getEdges.map(e => e.weight).sum
    val k = graph.getVertices.map(v => v -> graph.getOutEdges(v).map(e => e.weight).sum).toMap
    var sum = 0.0
    for (i <- graph.getVertices; j <- graph.getVertices) {
      if (clustering(i) == clustering(j)) {
        Option(graph.findEdge(i, j)) match {
          case Some(edge) =>
            sum += edge.weight - k(i) * k(j) / 2 / m
          case None =>
        }
      }
    }
    sum / 2 / m
  }
}
