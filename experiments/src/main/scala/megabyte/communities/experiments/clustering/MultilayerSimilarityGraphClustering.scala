package megabyte.communities.experiments.clustering

import java.io.File

import com.typesafe.scalalogging.Logger
import megabyte.communities.algo.graph.{MultilayerSpectralClustering, SpectralClustering}
import megabyte.communities.algo.points.XMeans
import megabyte.communities.experiments.config.ExperimentConfig.config._
import megabyte.communities.util.Graphs
import megabyte.communities.util.IO._
import org.jblas.DoubleMatrix

private class MultilayerSimilarityGraphClustering

object MultilayerSimilarityGraphClustering {

  private val LOG = Logger[MultilayerSimilarityGraphClustering]

  def main(args: Array[String]): Unit = {
    val u = subspace(10, 0.2)
    val clustering = XMeans.getClustering(u)
    logStats(clustering)
  }

  def subspace(k: Int, alpha: Double): DoubleMatrix = {
    val lSyms = networks.par.map { net =>
      readMatrixWithHeader(new File(similarityGraphsDir, net + ".csv"))._2
    }.seq.map(Graphs.symLaplacian)
    val us = networks.zip(lSyms)
      .map { case (net, l) =>
        val file = new File(subspaceDir, net + ".csv")
        readOrCalcMatrix(file) {
          SpectralClustering.toEigenspace(l)
        }
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
