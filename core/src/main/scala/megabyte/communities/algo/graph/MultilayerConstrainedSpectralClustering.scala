package megabyte.communities.algo.graph

import megabyte.communities.algo.points.KMeans
import megabyte.communities.util.Graphs._
import megabyte.communities.util.Matrices._
import org.jblas.DoubleMatrix

object MultilayerConstrainedSpectralClustering {

  def getClusteringIndConstraints(adjMatrices: Seq[DoubleMatrix], constraints: Seq[DoubleMatrix], alpha: Double): Seq[Int] = {
    val us = adjMatrices.zip(constraints).map { case (adj, q) =>
      normRowsI(ConstrainedSpectralClustering.toEigenspace(adj, q))
    }
    val lSyms = adjMatrices.map(symLaplacian)
    val n = adjMatrices.head.rows
    val lMod = new DoubleMatrix(n, n)
    lSyms.zip(us).foreach { case (li, ui) =>
      lMod.addi(
        li.subi(
          ui.mmul(ui.transpose()).muli(alpha)
        )
      )
    }
    val u = normRowsI(MultilayerSpectralClustering.toEigenspace(lMod, 1))
    KMeans.getClustering(u, 2)
  }
}
