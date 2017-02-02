package megabyte.communities.experiments

import java.io.File

import com.typesafe.scalalogging.Logger
import edu.uci.ics.jung.graph.Graph
import megabyte.communities.algo.graph.SpectralClustering
import megabyte.communities.entities.Edge
import megabyte.communities.util.DoubleMatrixOps._
import megabyte.communities.util.GraphFactory
import megabyte.communities.util.Graphs._
import megabyte.communities.util.Measures.modularity
import org.jblas.DoubleMatrix

import collection.JavaConversions._

private class SocialGraphsClustering

object SocialGraphsClustering {

  private val CITY = "Singapore"
  private val BASE_DIR = new File(s"experiments/src/main/resources/$CITY/graphs")
  private val LOG = Logger[SocialGraphsClustering]

  def main(args: Array[String]): Unit = {
    val graphs = Seq("twitter", "instagram", "foursquare")
      .map(name => readGraph(name + ".graphml"))
    val numeration = graphs.flatMap(_.getVertices).toSet.toList
    val n = numeration.size
    val adjs = graphs.map(g => symAdjacencyMatrix(applyNumeration(g, numeration), n))
    val summedAdj = adjs.fold(DoubleMatrix.zeros(n, n)) { (m1, m2) => m1 += m2 }
    val (k, clusteringSeq) = optimizeClustersCount(summedAdj, 2, 100)
    LOG.info("Best clustering:")
    logStats(summedAdj, k, clusteringSeq)
  }

  private def readGraph(fileName: String): Graph[String, Edge] = {
    GraphFactory.readGraph(new File(BASE_DIR, fileName))
  }

  private def optimizeClustersCount(adj: DoubleMatrix, start: Int, end: Int): (Int, Seq[Int]) = {
    (start to end)
      .map { k =>
        val clustering = SpectralClustering.getClustering(adj, k)
        logStats(adj, k, clustering)
        (k, clustering)
      }.maxBy { case (_, clustering) => modularity(adj, clustering) }
  }

  private def logStats(adj: DoubleMatrix, k: Int, clustering: Seq[Int]): Unit = {
    val modul = modularity(adj, clustering)
    val invClustering = clustering.groupBy(identity)
    val clustersNum = clustering.max + 1
    LOG.info(s"subspace dimensionality: $k")
    LOG.info(s"number of clusters: $clustersNum")
    LOG.info(s"sizes:" + invClustering.foldLeft("") { (s, cluster) => s"$s ${cluster._2.size}" })
    LOG.info(s"modularity: $modul")
  }
}
