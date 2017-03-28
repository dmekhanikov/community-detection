package megabyte.communities.experiments

import com.typesafe.scalalogging.Logger
import megabyte.communities.algo.graph.SpectralClustering
import megabyte.communities.util.Measures.modularity
import org.jblas.DoubleMatrix

private class Util

object Util {

  private val LOG = Logger[Util]

  def optimizeClustersCount(adj: DoubleMatrix, start: Int, end: Int): (Int, Seq[Int]) = {
    (start to end)
      .map { k =>
        val clustering = SpectralClustering.getClustering(adj, k)
        logStats(adj, k, clustering)
        (k, clustering)
      }.maxBy { case (_, clustering) => modularity(adj, clustering) }
  }

  def logStats(adj: DoubleMatrix, k: Int, clustering: Seq[Int]): Unit = {
    val modul = modularity(adj, clustering)
    val invClustering = clustering.groupBy(identity)
    val clustersNum = clustering.max + 1
    LOG.info(s"subspace dimensionality: $k")
    LOG.info(s"number of clusters: $clustersNum")
    LOG.info(s"sizes: " + (invClustering.map { cluster => cluster._2.size }.toSeq.sorted.reverse mkString " "))
    LOG.info(s"modularity: $modul")
  }
}
