package megabyte.communities.experiments

import java.io.File

import com.typesafe.scalalogging.Logger
import edu.uci.ics.jung.graph.Graph
import megabyte.communities.algo.graph.MultilayerSpectralClustering
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

  private implicit def pairToSeq[T](pair: (T, T)): Seq[T] = {
    Seq(pair._1, pair._2)
  }

  def main(args: Array[String]): Unit = {
    val graphs = Seq("twitter", "instagram", "foursquare")
      .map(name => readGraph(name + ".graphml"))
    val numeration = graphs.flatMap(_.getVertices).toSet.toList
    val adjs = graphs.flatMap(g => symAdj(g, numeration))
    val clusteringSeq = MultilayerSpectralClustering.getClustering(adjs, 2, 0.1)
    val invClustering = clusteringSeq.groupBy(Predef.identity)
    LOG.info(s"clusters count: ${invClustering.size}")
    LOG.info(s"sizes: ${invClustering(0)}, ${invClustering(1)}")
  }

  private def readGraph(fileName: String): Graph[String, Edge] = {
    GraphFactory.readGraph(new File(BASE_DIR, fileName))
  }

  private def symAdj[V](graph: Graph[V, Edge], numeration: Seq[V]): (DoubleMatrix, DoubleMatrix) = {
    val numeratedGraph = applyNumeration(graph, numeration)
    val a = adjacencyMatrix(numeratedGraph)
    val aT = a.transpose()
    val a1 = a.mul(aT)
    val a2 = aT.mul(a)
    (a1, a2)
  }
}
