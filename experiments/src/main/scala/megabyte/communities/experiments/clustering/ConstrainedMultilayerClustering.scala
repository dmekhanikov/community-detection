package megabyte.communities.experiments.clustering

import com.typesafe.scalalogging.Logger
import megabyte.communities.algo.graph.{LuConstrainedSpectralClustering, MultilayerSpectralClustering}
import megabyte.communities.algo.points.KMeans
import megabyte.communities.experiments.config.ExperimentConfig.config._
import megabyte.communities.experiments.util.DataUtil._
import megabyte.communities.util.Graphs.symLaplacian

object ConstrainedMultilayerClustering {

  private val LOG = Logger[ConstrainedMultilayerClustering.type]

  private val k = 2
  private val knn = 10
  private val alpha = 0.2

  def main(args: Array[String]): Unit = {
    LOG.info("Reading adjacency matrices")
    val (networksHashes, adjs) = networks.par.map(readAdj).seq.unzip
    val lSyms = adjs.par.map(symLaplacian).seq
    LOG.info("Reading constraint graphs")
    val constraints = networks.zip(networksHashes).par.map { case (net, hashes) =>
      readConstraintsMatrix(s"$net.graphml", hashes)
    }.seq

    LOG.info("Calculating subspace representations for each layer with applied constraints")
    val us = networks.zip(adjs).zip(constraints).par
      .map { case ((net, adj), q) =>
        LuConstrainedSpectralClustering.toEigenspace(adj, q, knn, alpha)
      }.seq
    val u = MultilayerSpectralClustering.toCommonEigenspace(lSyms, us, k, alpha)

    val clustering = KMeans.getClustering(u, k)
    logStats(clustering)
  }

  private def logStats(clustering: Seq[Int]): Unit = {
    val invClustering = clustering.groupBy(identity)
    val clustersNum = clustering.max + 1
    LOG.info(s"number of clusters: $clustersNum")
    LOG.info(s"sizes:" + invClustering.foldLeft("") { (s, cluster) => s"$s ${cluster._2.size}" })
  }
}
