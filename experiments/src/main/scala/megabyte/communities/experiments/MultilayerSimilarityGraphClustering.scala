package megabyte.communities.experiments

import java.io.File

import com.typesafe.scalalogging.Logger
import megabyte.communities.algo.graph.MultilayerSpectralClustering
import megabyte.communities.experiments.ClusteringUtil._
import megabyte.communities.experiments.config.ExperimentConfig

private class MultilayerSimilarityGraphClustering

object MultilayerSimilarityGraphClustering {

  private val LOG = Logger[MultilayerSimilarityGraphClustering]

  private val BASE_DIR = ExperimentConfig.config.baseDir
  private val CITY = ExperimentConfig.config.city
  private val GRAPHS_DIR = new File(s"$BASE_DIR/$CITY/graphs/similarity")

  private val INPUT_FILES = Seq(
    "foursquare.csv",
    "twitter.csv",
    "instagram.csv")

  def main(args: Array[String]): Unit = {
    val adjs = INPUT_FILES.par.map { fileName =>
      readDataFile(new File(GRAPHS_DIR, fileName))._2
    }.seq
    val clustering = MultilayerSpectralClustering.getClustering(adjs, 10, 0.2)
    logStats(clustering)
  }

  private def logStats(clustering: Seq[Int]): Unit = {
    val invClustering = clustering.groupBy(identity)
    val clustersNum = clustering.max + 1
    LOG.info(s"number of clusters: $clustersNum")
    LOG.info(s"sizes:" + invClustering.foldLeft("") { (s, cluster) => s"$s ${cluster._2.size}" })
  }
}
