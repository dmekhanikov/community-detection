package megabyte.communities

import edu.uci.ics.jung.graph.{Graph, UndirectedSparseGraph}
import org.slf4j.LoggerFactory

import scala.collection.JavaConversions._
import scala.collection.{Map, mutable}

object Louvain {

  private val MAX_ITERATIONS = 20
  private val LOG = LoggerFactory.getLogger(Louvain.getClass)

  // yeah, I feel bad for this, sorry
  private var m = 0.0

  def getClustering(graph: Graph[Long, Edge]): Map[Long, Int] = {
    val dendrogram = getDendrogram(graph)
    val clustering = mutable.Map(dendrogram.layers.last.toSeq:_*)
    for (layer <- dendrogram.layers.reverse.tail) {
      clustering.foreach { case (v, c) => clustering(v) = layer(c) }
    }
    clustering
  }

  def getDendrogram(graph: Graph[Long, Edge]): Dendrogram = {
    LOG.debug("Getting a dendrogram")
    m = totalWeight(graph.getEdges)
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
    LOG.debug("Getting a local clustering")
    val clustering = initClustering(graph)
    var change = true
    var it = 0
    while(it < MAX_ITERATIONS && change) {
      LOG.debug("Starting a getLocalClustering loop iteration")
      it += 1
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
    LOG.debug("Collapsing clusters to nodes")
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

  private def modularityGain(v: Long,
                             cluster: Int,
                             graph: Graph[Long, Edge],
                             clustering: mutable.Map[Long, Int]): Double = {
    val vCluster = clustering(v)
    clustering(v) = -1
    val clusterVertices = graph.getVertices.filter(v => clustering(v) == cluster).toSet

    // cluster's inner edges
    val inSum = totalWeight(clusterVertices
      .flatMap(v => graph.getOutEdges(v))
      .filter(e => graph.getEndpoints(e).forall(v => clustering(v) == cluster)).toSeq) / 2
    // edges incident to the cluster
    val totSum = totalWeight(clusterVertices.flatMap(v => graph.getOutEdges(v)).toSeq) - inSum
    // edges from v to the cluster
    val kvin = totalWeight(graph.getOutEdges(v)
      .filter(e => graph.getEndpoints(e).exists(node => clusterVertices.contains(node))))
    // edges incident to v
    val kv = totalWeight(graph.getOutEdges(v))

    clustering(v) = vCluster
    ((inSum + kvin) / 2 / m - math.pow((totSum + kv) / 2 / m, 2)) -
      (inSum / 2 / m - math.pow(totSum / 2 / m, 2) - math.pow(kv / 2 / m, 2))
  }

  private def totalWeight(edges: Iterable[Edge]): Double = {
    edges.map(e => e.weight).sum
  }

  private def initClustering(graph: Graph[Long, Edge]): mutable.Map[Long, Int] = {
    val clustering = mutable.Map[Long, Int]()
    graph.getVertices.zipWithIndex.foreach { case (v, i) =>
      clustering(v) = i + 1
    }
    clustering
  }
}
