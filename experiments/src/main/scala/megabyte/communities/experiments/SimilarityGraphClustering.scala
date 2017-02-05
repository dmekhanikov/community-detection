package megabyte.communities.experiments

import java.io.File

import com.typesafe.scalalogging.Logger
import megabyte.communities.experiments.ClusteringUtil._
import megabyte.communities.experiments.config.ExperimentConfig

private class SimilarityGraphClustering

object SimilarityGraphClustering {

  private val BASE_DIR = ExperimentConfig.config.baseDir
  private val CITY = ExperimentConfig.config.city
  private val NETWORK = ExperimentConfig.config.network.get
  private val ADJ_FILE = new File(s"$BASE_DIR/$CITY/graphs/similarity/$NETWORK.csv")
  private val LOG = Logger[SimilarityGraphClustering]

  def main(args: Array[String]): Unit = {
    val adj = readDataFile(ADJ_FILE)
    val (k, clusteringSeq) = optimizeClustersCount(adj, 2, 100)
    LOG.info("Best clustering:")
    logStats(adj, k, clusteringSeq)
  }
}
