package megabyte.communities.algo.graph

import com.typesafe.scalalogging.Logger
import edu.uci.ics.jung.graph.Graph
import megabyte.communities.algo.points.XMeans
import megabyte.communities.entities.Edge
import megabyte.communities.util.DoubleMatrixOps._
import megabyte.communities.util.Graphs._
import org.jblas.{DoubleMatrix, Eigen}

private class MultilayerSpectralClustering

object MultilayerSpectralClustering {

  private val LOG = Logger[MultilayerSpectralClustering]

  // alpha - coefficient in the objective function. Importance of layers to be close on the manifold.
  def getClustering(graphs: Seq[Graph[Long, Edge]], k: Int, alpha: Double): Map[Long, Int] = {
    val numeration = numerateNodes(graphs.head)
    val adjMatrices = graphs.par.map { g =>
      val renumG = applyNumeration(g, numeration)
      adjacencyMatrix(renumG)
    }
    val renumClustering = getClustering(adjMatrices.seq, k, alpha)
    Map(numeration.zip(renumClustering): _*)
  }

  def getClustering(adjMatrices: Seq[DoubleMatrix], dim: Int, alpha: Double): Seq[Int] = {
    LOG.info(s"Starting multilayer clustering computation")
    val lSyms = adjMatrices.map(symLaplacian)
    val us = lSyms.zipWithIndex.map { case (l, i) =>
      LOG.info(s"Starting processing layer #${i + 1}")
      val u = toEigenspace(l).prefixColumns(dim)
      LOG.info(s"${i + 1} / ${adjMatrices.size} layers processed")
      u
    }
    val u = toCommonEigenspace(lSyms, us, dim, alpha)
    LOG.info("Running a final step clustering")
    XMeans.getClustering(u)
  }

  def getLMod(us: Seq[DoubleMatrix], lSyms: Seq[DoubleMatrix], dim: Int, alpha: Double): DoubleMatrix = {
    LOG.info("Building modified Laplacian matrix")
    val n = lSyms.head.rows
    val lMod = new DoubleMatrix(n, n)
    lSyms.zip(us).foreach { case (li, ui) =>
      lMod += (li -= ((ui * ui.transpose()) *= alpha))
    }
    lMod
  }

  def toCommonEigenspace(us: Seq[DoubleMatrix], lSyms: Seq[DoubleMatrix], dim: Int, alpha: Double): DoubleMatrix = {
    val lMod = getLMod(us, lSyms, dim, alpha)
    LOG.info("Searching for eigenvectors of the modified Laplacian matrix")
    toEigenspace(lMod).prefixColumns(dim).normRowsI()
  }

  def toEigenspace(matrix: DoubleMatrix): DoubleMatrix = {
    val Array(vectors, valuesMatrix) = Eigen.symmetricEigenvectors(matrix)
    val indices = valuesMatrix
      .diagonalElements()
      .zipWithIndex
      .sortBy(_._1)
      .map(_._2)
      .toArray
    vectors.getColumns(indices)
  }
}
