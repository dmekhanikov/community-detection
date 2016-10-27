package megabyte.communities.util

import edu.uci.ics.jung.graph.Graph
import megabyte.communities.entities.Edge

import scala.collection.JavaConversions._
import scala.collection.Map

object Metrics {

  def modularity(graph: Graph[Long, Edge], clustering: Map[Long, Int]): Double = {
    val n = graph.getVertexCount
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
