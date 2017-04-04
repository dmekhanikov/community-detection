package megabyte.communities.experiments.clustering

import java.io.File

import com.typesafe.scalalogging.Logger
import megabyte.communities.experiments.util.ClusteringUtil._
import megabyte.communities.experiments.config.ExperimentConfig.config._
import megabyte.communities.util.IO._

object SimilarityGraphClustering {

  private val LOG = Logger[SimilarityGraphClustering.type]

  private val adjFile = new File(s"$baseDir/$city/graphs/similarity/$network.csv")

  def main(args: Array[String]): Unit = {
    val (_, adj) = readMatrixWithHeader(adjFile)
    val (k, clusteringSeq) = optimizeClustersCount(adj, 2, 100)
    LOG.info("Best clustering:")
    logStats(adj, k, clusteringSeq)
  }
}
