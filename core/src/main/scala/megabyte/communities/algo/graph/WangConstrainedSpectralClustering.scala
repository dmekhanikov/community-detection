package megabyte.communities.algo.graph

import megabyte.communities.algo.points.KMeans
import megabyte.communities.util.DoubleMatrixOps._
import megabyte.communities.util.Eigen._
import megabyte.communities.util.Graphs._
import org.jblas.DoubleMatrix
import org.jblas.DoubleMatrix._
import org.jblas.Eigen._

object WangConstrainedSpectralClustering {

  def getClustering(adj: DoubleMatrix, constraints: DoubleMatrix, k: Int): Seq[Int] = {
    val u = toEigenspace(adj, constraints).prefixColumns(k)
    KMeans.getClustering(u, k)
  }

  def toEigenspace(adj: DoubleMatrix, constraints: DoubleMatrix): DoubleMatrix = {
    val n = adj.columns
    val vol = adj.sum
    val dNorm = degreeMatrix(adj).sqrtDiagI().invDiagI()
    val lNorm = symLaplacian(adj)
    val qNorm = (dNorm \* constraints) *\= dNorm
    val maxQLam = symmetricEigenvalues(qNorm).sort().get(n - 1)

    // solve generalized eigenvalue problem
    val q1 = qNorm -= (eye(n) *= (maxQLam * 0.5))
    val (vectors, values) = generalizedEigenvectors(lNorm, q1)
    // normalize eigenvectors
    for (i <- 0 until vectors.columns) {
      val col = vectors.getColumn(i)
      vectors.putColumn(i, (col /= col.norm) *= math.pow(vol, 0.5))
    }

    val I = (0 until n).filter(i => values.get(i) >= 0)
    val feasibleVectors = vectors.getColumns(I.toArray)

    val costs = (0 until feasibleVectors.columns).map { i =>
      val v = feasibleVectors.getColumn(i)
      (v.transpose() * lNorm * v).get(0)
    }
    val indices = costs.zipWithIndex
      .sorted
      .map(_._2)
    dNorm * feasibleVectors.getColumns(indices.toArray)
  }
}
