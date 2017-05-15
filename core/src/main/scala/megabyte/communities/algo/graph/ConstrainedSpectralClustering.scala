package megabyte.communities.algo.graph

import megabyte.communities.algo.constraints.ConstraintsApplier
import megabyte.communities.algo.points.KMeans
import megabyte.communities.util.DoubleMatrixOps._
import megabyte.communities.util.Graphs.symLaplacian
import org.jblas.DoubleMatrix

class ConstrainedSpectralClustering(private val constraintsApplier: ConstraintsApplier) {

  def getClustering(w: DoubleMatrix, q: DoubleMatrix, k: Int): Seq[Int] = {
    val u = toEigenspace(w, q).prefixColumns(k)
    KMeans.getClustering(u, k)
  }

  def toEigenspace(w: DoubleMatrix, q: DoubleMatrix): DoubleMatrix = {
    val wMod = constraintsApplier.applyConstraints(w, q)
    val lMod = symLaplacian(wMod)
    SpectralClustering.toEigenspace(lMod)
  }
}
