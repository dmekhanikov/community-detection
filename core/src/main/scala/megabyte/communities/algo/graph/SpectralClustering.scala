package megabyte.communities.algo.graph

import edu.uci.ics.jung.graph.Graph
import megabyte.communities.algo.points.XMeans
import megabyte.communities.entities.Edge
import megabyte.communities.util.Graphs._
import org.jblas.DoubleMatrix
import org.jblas.Eigen._

object SpectralClustering {

  def getClustering(graph: Graph[Long, Edge], k: Int): Map[Long, Int] = {
    val numeration = numerateNodes(graph)
    val renumeratedGraph = applyNumeration(graph, numeration)
    val adj = adjacencyMatrix(renumeratedGraph)
    val renumeratedClustering = getClustering(adj, k)
    Map(numeration.zip(renumeratedClustering):_*)
  }

  def getClustering(adj: DoubleMatrix, k: Int): Seq[Int] = {
    val d = degreeMatrix(adj)
    val l = d.sub(adj)
    val Array(vectors, values) = symmetricGeneralizedEigenvectors(l, d)
    // sort by values and take first k
    val indices = values.toArray
      .zipWithIndex
      .sortBy(_._1)
      .take(k)
      .map(_._2)
    val clippedVectors = vectors.getColumns(indices)
    XMeans.getClustering(clippedVectors)
  }
}
