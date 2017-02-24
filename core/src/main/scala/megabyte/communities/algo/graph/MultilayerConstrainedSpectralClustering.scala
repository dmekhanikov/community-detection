package megabyte.communities.algo.graph

import com.typesafe.scalalogging.Logger
import megabyte.communities.algo.points.KMeans
import megabyte.communities.util.DoubleMatrixOps._
import megabyte.communities.util.Graphs._
import org.jblas.DoubleMatrix

private class MultilayerConstrainedSpectralClustering

object MultilayerConstrainedSpectralClustering {

  private val LOG = Logger[MultilayerConstrainedSpectralClustering]

  def getClustering(adjMatrices: Seq[DoubleMatrix],
                    constraints: Seq[DoubleMatrix],
                    alpha: Double): Seq[Int] = {
    val u = toEigenspace(adjMatrices, constraints, alpha)
    KMeans.getClustering(u, 2)
  }

  def toEigenspace(adjMatrices: Seq[DoubleMatrix],
                   constraints: Seq[DoubleMatrix],
                   alpha: Double): DoubleMatrix = {
    LOG.info("Processing constraints")
    val us = adjMatrices.zip(constraints).zipWithIndex.map { case ((adj, q), i) =>
      val u = ConstrainedSpectralClustering.toEigenspace(adj, q)
      LOG.info(s"Processed constraints on layer #${i + 1}/${adjMatrices.size}")
      u
    }
    val lSyms = adjMatrices.map(symLaplacian)
    val n = adjMatrices.head.rows
    val lMod = new DoubleMatrix(n, n)
    LOG.info("Calculating Laplacian matrix for multilayer constrained clustering")
    lSyms.zip(us).foreach { case (li, ui) =>
      lMod += (li -= ((ui * ui.transpose()) *= alpha))
    }
    MultilayerSpectralClustering.toEigenspace(lMod, 1).normRowsI()
  }
}
