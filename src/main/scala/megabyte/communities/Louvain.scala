package megabyte.communities

import edu.uci.ics.jung.graph.{Graph, UndirectedSparseGraph}

import scala.collection.JavaConversions._
import scala.collection.{Map, mutable}

object Louvain {

  def getDendrogram(graph: Graph[Long, Edge]): Dendrogram = {
    def makeDendrogram(g: Graph[Long, Edge], layers: List[Map[Long, Int]]) : List[Map[Long, Int]] = {
      val localClustering = getLocalClustering(g)
      val anyChange = localClustering.keys
        .groupBy(v => localClustering(v)).values
        .exists(c => c.size > 1)
      if (anyChange) {
        val shrunkGraph = shrinkGraph(g, localClustering)
        makeDendrogram(shrunkGraph, localClustering :: layers)
      } else {
        layers
      }
    }
    val dendrogram = new Dendrogram
    dendrogram.layers = makeDendrogram(graph, List())
    dendrogram
  }

  private def getLocalClustering(graph: Graph[Long, Edge]): Map[Long, Int] = {
    val clustering = initClustering(graph)
    var change = true
    while(change) {
      change = false
      for (v <- graph.getVertices) {
        val neighboringClusters = graph
          .getNeighbors(v)
          .map(node => clustering(node))
          .filter(c => c != clustering(v))
        if (neighboringClusters.nonEmpty) {
          val bestCluster = neighboringClusters.maxBy { candidateCluster =>
            modularityGain(v, clustering(v), candidateCluster, graph, clustering)
          }
          val maxGain = modularityGain(v, clustering(v), bestCluster, graph, clustering)
          if (maxGain > 0) {
            clustering(v) = bestCluster
            change = true
          }
        }
      }
    }
    clustering
  }

  private def shrinkGraph(graph: Graph[Long, Edge], clustering: Map[Long, Int]): Graph[Long, Edge] = {
    val clusters = clustering.keys.groupBy(v => clustering(v))
    val newGraph = new UndirectedSparseGraph[Long, Edge]()
    clusters.foreach { case (cluster, _) => newGraph.addVertex(cluster) }
    graph.getEdges.foreach { e =>
      // TODO rewrite in Scala style
      val endpoints = graph.getEndpoints(e)
      val from = clustering(endpoints.getFirst)
      val to = clustering(endpoints.getSecond)
      Option(newGraph.findEdge(from, to)) match {
        case Some(edge) =>
          edge.weight += e.weight
        case None =>
          val edge = new Edge(e.weight)
          newGraph.addEdge(edge, from, to)
      }
    }
    newGraph
  }

  private def modularityGain(v: Long,
                             from: Int,
                             to: Int,
                             graph: Graph[Long, Edge],
                             clustering: mutable.Map[Long, Int]): Double = {
    val leaveGain = -modularityGain(v, from, graph, clustering)
    val enterGain = modularityGain(v, to, graph, clustering)
    leaveGain + enterGain
  }

  implicit class GraphExtended(graph: Graph[Long, Edge]) {
    def subGraphWeight(pred: Edge => Boolean): Double =
      graph.getEdges.filter(pred).map(e => e.weight).sum
  }

  private def modularityGain(v: Long,
                             cluster: Int,
                             graph: Graph[Long, Edge],
                             clustering: mutable.Map[Long, Int]): Double = {
    val vCluster = clustering(v)
    clustering(v) = -1

    // all edges in graph
    val m = graph.subGraphWeight {e => true }
    // cluster's inner edges
    val inSum = graph.subGraphWeight { e => graph.getEndpoints(e).forall(v => clustering(v) == cluster) }
    // edges incident to the cluster
    val totSum = graph.subGraphWeight { e => graph.getEndpoints(e).exists(v => clustering(v) == cluster) }
    // edges from v to the cluster
    val kvin = graph.subGraphWeight { e =>
      val endpoints = graph.getEndpoints(e)
      endpoints.contains(v) &&
        endpoints.map(v => clustering(v)).contains(cluster)
    }
    // edges incident to v
    val kv = graph.subGraphWeight(e => graph.getEndpoints(e).contains(v))

    clustering(v) = vCluster
    ((inSum + kvin) / 2 / m - math.pow((totSum + kv) / 2 / m, 2)) -
      (inSum / 2 / m - math.pow(totSum / 2 / m, 2) - math.pow(kv / 2 / m, 2))
  }

  private def initClustering(graph: Graph[Long, Edge]): mutable.Map[Long, Int] = {
    val clustering = mutable.Map[Long, Int]()
    graph.getVertices.zipWithIndex.foreach { case (v, i) =>
      clustering(v) = i + 1
    }
    clustering
  }
}
