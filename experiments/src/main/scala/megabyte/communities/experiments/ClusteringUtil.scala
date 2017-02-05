package megabyte.communities.experiments

import java.io.File

import com.typesafe.scalalogging.Logger
import megabyte.communities.algo.graph.SpectralClustering
import megabyte.communities.util.Measures.modularity
import org.jblas.DoubleMatrix

private class ClusteringUtil

object ClusteringUtil {

  private val LOG = Logger[ClusteringUtil]

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

  def readDataFile(file: File): DoubleMatrix = {
    LOG.info(s"Reading data from file $file")
    val source = io.Source.fromFile(file)
    val data = source.getLines.drop(1).map { line =>
      line.split(",").map(_.trim).map(_.toDouble).array
    }.toArray
    new DoubleMatrix(data)
  }
}
