package megabyte.communities.algo.graph

import megabyte.communities.algo.points.KMeans
import megabyte.communities.util.DoubleMatrixOps._
import megabyte.communities.util.Graphs._
import org.jblas.DoubleMatrix

object MultilayerConstrainedSpectralClustering {

  def getClusteringIndConstraints(adjMatrices: Seq[DoubleMatrix], constraints: Seq[DoubleMatrix], alpha: Double): Seq[Int] = {
    val us = adjMatrices.zip(constraints).map { case (adj, q) =>
      ConstrainedSpectralClustering.toEigenspace(adj, q).normRowsI()
    }
    val lSyms = adjMatrices.map(symLaplacian)
    val n = adjMatrices.head.rows
    val lMod = new DoubleMatrix(n, n)
    lSyms.zip(us).foreach { case (li, ui) =>
      lMod += (li -= ((ui * ui.transpose()) *= alpha))
    }
    val u = MultilayerSpectralClustering.toEigenspace(lMod, 1).normRowsI()
    KMeans.getClustering(u, 2)
  }
}
