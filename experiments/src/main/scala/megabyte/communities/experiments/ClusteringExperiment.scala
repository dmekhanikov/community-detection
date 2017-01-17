package megabyte.communities.experiments

import java.io.File

import edu.uci.ics.jung.graph.Graph
import megabyte.communities.algo.graph.MultilayerSpectralClustering
import megabyte.communities.entities.Edge
import megabyte.communities.util.GraphFactory
import megabyte.communities.util.Graphs._
import org.jblas.DoubleMatrix

object ClusteringExperiment {

  private val CITY = "Singapore"
  private val BASE_DIR = new File(s"experiments/src/main/resources/$CITY/graphs")

  private implicit def pairToSeq[T](pair: (T, T)): Seq[T] = {
    Seq(pair._1, pair._2)
  }

  def main(args: Array[String]): Unit = {
    val adjs = Seq("twitter", "instagram", "foursquare")
      .map(name => readGraph(name + ".graphml")._1)
      .flatMap(symAdj)
    val clusteringSeq = MultilayerSpectralClustering.getClustering(adjs, 2, 0.1)
    val invClustering = clusteringSeq.groupBy(i => i)
    print(s"sizes: ${invClustering(0)}, ${invClustering(1)}")
  }

  private def readGraph(fileName: String): (Graph[Int, Edge], Seq[String]) = {
    val graph = GraphFactory.readGraph(new File(BASE_DIR, fileName))
    val numeration = numerateNodes(graph)
    (applyNumeration(graph, numeration), numeration)
  }

  private def symAdj(graph: Graph[Int, Edge]): (DoubleMatrix, DoubleMatrix) = {
    val a = adjacencyMatrix(graph)
    val aT = a.transpose()
    val a1 = a.mul(aT)
    val a2 = aT.mul(a)
    (a1, a2)
  }
}
