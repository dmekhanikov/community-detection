package megabyte.communities.experiments.clustering

import java.io.File

import com.typesafe.scalalogging.Logger
import megabyte.communities.experiments.clustering.ClusteringUtil._
import megabyte.communities.experiments.config.ExperimentConfig
import megabyte.communities.util.DoubleMatrixOps._
import megabyte.communities.util.GraphFactory
import megabyte.communities.util.Graphs._
import org.jblas.DoubleMatrix

import collection.JavaConversions._

private class SocialGraphsClustering

object SocialGraphsClustering {

  private val CITY = ExperimentConfig.config.city
  private val BASE_DIR = ExperimentConfig.config.baseDir
  private val GRAPHS_DIR = new File(s"$BASE_DIR/$CITY/graphs/connections")
  private val LOG = Logger[SocialGraphsClustering]

  def main(args: Array[String]): Unit = {
    val graphs = Seq("twitter", "instagram", "foursquare")
      .map(name => GraphFactory.readGraph(new File(GRAPHS_DIR, name)))
    val numeration = graphs.flatMap(_.getVertices).toSet.toList
    val n = numeration.size
    val adjs = graphs.map(g => symAdjacencyMatrix(applyNumeration(g, numeration), n))
    val summedAdj = adjs.fold(DoubleMatrix.zeros(n, n)) { (m1, m2) => m1 += m2 }
    val (k, clusteringSeq) = optimizeClustersCount(summedAdj, 2, 100)
    LOG.info("Best clustering:")
    logStats(summedAdj, k, clusteringSeq)
  }
}
