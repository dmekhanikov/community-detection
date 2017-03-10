package megabyte.communities.experiments.clustering

import java.io.File

import com.typesafe.scalalogging.Logger
import megabyte.communities.algo.graph.MultilayerSpectralClustering
import megabyte.communities.algo.points.XMeans
import megabyte.communities.experiments.clustering.ClusteringUtil._
import megabyte.communities.experiments.config.ExperimentConfig
import megabyte.communities.util.Graphs
import megabyte.communities.util.DoubleMatrixOps._

private class MultilayerSimilarityGraphClustering

object MultilayerSimilarityGraphClustering {

  private val LOG = Logger[MultilayerSimilarityGraphClustering]

  private val BASE_DIR = ExperimentConfig.config.baseDir
  private val CITY = ExperimentConfig.config.city
  private val GRAPHS_DIR = new File(s"$BASE_DIR/$CITY/graphs/similarity")
  private val SUBSPACE_DIR = new File(BASE_DIR, s"$CITY/subspaces/sym")

  private val INPUT_FILES = Seq(
    "foursquare.csv",
    "twitter.csv",
    "instagram.csv")

  private val k = 10
  private val alpha = 0.2

  def main(args: Array[String]): Unit = {
    val adjs = INPUT_FILES.par.map { fileName =>
      readDataFile(new File(GRAPHS_DIR, fileName))._2
    }.seq
    val lSyms = adjs.map(Graphs.symLaplacian)
    val us = INPUT_FILES.zip(lSyms)
      .map { case (fileName, l) =>
        val file = new File(SUBSPACE_DIR, fileName)
        readOrCalcMatrix(file) {
          MultilayerSpectralClustering.toEigenspace(l)
        }.prefixColumns(k)
      }
    val u = MultilayerSpectralClustering.toCommonEigenspace(us, lSyms, k, alpha)
    val clustering = XMeans.getClustering(u)
    logStats(clustering)
  }

  private def logStats(clustering: Seq[Int]): Unit = {
    val invClustering = clustering.groupBy(identity)
    val clustersNum = clustering.max + 1
    LOG.info(s"number of clusters: $clustersNum")
    LOG.info(s"sizes:" + invClustering.foldLeft("") { (s, cluster) => s"$s ${cluster._2.size}" })
  }
}
