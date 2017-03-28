package megabyte.communities.experiments.clustering

import java.io.File

import com.typesafe.scalalogging.Logger
import megabyte.communities.algo.graph.MultilayerSpectralClustering
import megabyte.communities.algo.points.XMeans
import megabyte.communities.experiments.config.ExperimentConfig.config._
import megabyte.communities.util.DoubleMatrixOps._
import megabyte.communities.util.Graphs
import megabyte.communities.util.IO._
import org.jblas.DoubleMatrix

private class MultilayerSimilarityGraphClustering

object MultilayerSimilarityGraphClustering {

  private val LOG = Logger[MultilayerSimilarityGraphClustering]

  private val INPUT_FILES = Seq(
    "foursquare.csv",
    "twitter.csv",
    "instagram.csv")

  private val k = 10
  private val alpha = 0.2

  def main(args: Array[String]): Unit = {
    val u = subspace()
    val clustering = XMeans.getClustering(u)
    logStats(clustering)
  }

  def subspace(): DoubleMatrix = {
    val adjs = INPUT_FILES.par.map { fileName =>
      readDataFile(new File(similarityGraphsDir, fileName))._2
    }.seq
    val lSyms = adjs.map(Graphs.symLaplacian)
    val us = INPUT_FILES.zip(lSyms)
      .map { case (fileName, l) =>
        val file = new File(subspaceDir, fileName)
        readOrCalcMatrix(file) {
          MultilayerSpectralClustering.toEigenspace(l)
        }.prefixColumns(k)
      }
    MultilayerSpectralClustering.toCommonEigenspace(us, lSyms, k, alpha)
  }

  private def logStats(clustering: Seq[Int]): Unit = {
    val invClustering = clustering.groupBy(identity)
    val clustersNum = clustering.max + 1
    LOG.info(s"number of clusters: $clustersNum")
    LOG.info(s"sizes:" + invClustering.foldLeft("") { (s, cluster) => s"$s ${cluster._2.size}" })
  }
}
