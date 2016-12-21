package megabyte.communities.algo.graph

import edu.uci.ics.jung.graph.Graph
import megabyte.communities.algo.points.KMeans
import megabyte.communities.entities.Edge
import megabyte.communities.util.Graphs._
import megabyte.communities.util.Matrices._
import org.jblas.{DoubleMatrix, Eigen}

object MultilayerSpectralClustering {

  // alpha - coefficient in the objective function. Importance of layers to be close on the manifold.
  def getClustering(graphs: Seq[Graph[Long, Edge]], k: Int, alpha: Double): Map[Long, Int] = {
    val numeration = numerateNodes(graphs.head)
    val adjMatrices = graphs.map { g =>
      val renumG = applyNumeration(g, numeration)
      adjacencyMatrix(renumG)
    }
    val renumClustering = getClustering(adjMatrices, k, alpha)
    Map(numeration.zip(renumClustering):_*)
  }

  def getClustering(adjMatrices: Seq[DoubleMatrix], k: Int, alpha: Double): Seq[Int] = {
    val lSyms = adjMatrices.map(symLaplacian)
    val us = lSyms.map(l => normRowsI(toEigenspace(l, k)))
    val n = adjMatrices.head.rows
    val lMod = new DoubleMatrix(n, n)
    lSyms.zip(us).foreach { case (li, ui) =>
        lMod.addi(
          li.subi(
            ui.mmul(ui.transpose()).muli(alpha)
          )
        )
    }
    val u = normRowsI(toEigenspace(lMod, k))
    KMeans.getClustering(u, k)
  }

  def toEigenspace(matrix: DoubleMatrix, dim: Int): DoubleMatrix = {
    val Array(vectors, values) = Eigen.symmetricEigenvectors(matrix)
    // sort by values and take first k
    val indices = (0 until values.columns)
      .map(i => matrix.get(i, i))
      .zipWithIndex
      .sortBy(_._1)
      .take(dim)
      .map(_._2)
      .toArray
    vectors.getColumns(indices)
  }
}
