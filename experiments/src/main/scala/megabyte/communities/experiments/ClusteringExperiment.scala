package megabyte.communities.experiments

import java.io.File

import com.typesafe.scalalogging.Logger
import edu.uci.ics.jung.graph.Graph
import megabyte.communities.algo.graph.SpectralClustering
import megabyte.communities.entities.Edge
import megabyte.communities.util.GraphFactory
import megabyte.communities.util.Graphs._
import org.jblas.DoubleMatrix

import collection.JavaConversions._

private class ClusteringExperiment

object ClusteringExperiment {

  private val CITY = "Singapore"
  private val BASE_DIR = new File(s"experiments/src/main/resources/$CITY/graphs")
  private val LOG = Logger[ClusteringExperiment]

  def main(args: Array[String]): Unit = {
    val graphs = Seq("twitter", "instagram", "foursquare")
      .map(name => readGraph(name + ".graphml"))
    val numeration = graphs.flatMap(_.getVertices).toSet.toList
    val n = numeration.size
    val adjs = graphs.map(g => symAdjacencyMatrix(applyNumeration(g, numeration), n))
    val summedAdj = makeAdjBinary(adjs.fold(DoubleMatrix.zeros(n, n)) { (m1, m2) => m1.addi(m2) })
    val clusteringSeq = SpectralClustering.getClustering(summedAdj, 5)
    val invClustering = clusteringSeq.groupBy(Predef.identity)
    LOG.info(s"clusters count: ${invClustering.size}")
    LOG.info(s"sizes:" + invClustering.foldLeft("") {(s, cluster) => s"$s ${cluster._2.size}"})
  }

  private def readGraph(fileName: String): Graph[String, Edge] = {
    GraphFactory.readGraph(new File(BASE_DIR, fileName))
  }
}
