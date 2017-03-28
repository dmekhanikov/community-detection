package megabyte.communities.experiments.clustering

import java.io.File

import com.typesafe.scalalogging.Logger
import megabyte.communities.experiments.Util._
import megabyte.communities.experiments.config.ExperimentConfig.config._
import megabyte.communities.util.DoubleMatrixOps._
import megabyte.communities.util.GraphFactory
import megabyte.communities.util.Graphs._
import org.jblas.DoubleMatrix

import collection.JavaConversions._

private class SocialGraphsClustering

object SocialGraphsClustering {

  private val LOG = Logger[SocialGraphsClustering]

  def main(args: Array[String]): Unit = {
    val graphs = Seq("twitter", "instagram", "foursquare")
      .map(name => GraphFactory.readGraph(new File(similarityGraphsDir, s"$name.graphml")))
    val numeration = graphs.flatMap(_.getVertices).toSet.toList
    val n = numeration.size
    val adjs = graphs.map(g => symAdjacencyMatrix(applyNumeration(g, numeration), n))
    val summedAdj = adjs.fold(DoubleMatrix.zeros(n, n)) { (m1, m2) => m1 += m2 }
    val (k, clusteringSeq) = optimizeClustersCount(summedAdj, 2, 100)
    LOG.info("Best clustering:")
    logStats(summedAdj, k, clusteringSeq)
  }
}
