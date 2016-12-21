package megabyte.communities.algo.graph

import megabyte.communities.util.Graphs._
import megabyte.communities.util.Measures._
import org.jblas.DoubleMatrix
import org.jblas.DoubleMatrix._
import org.jblas.Eigen._
import megabyte.communities.util.Eigen._
import org.jblas.MatrixFunctions._
import org.jblas.Solve._

object ConstrainedSpectralClustering {

  def getClustering(adj: DoubleMatrix, constraints: DoubleMatrix): Seq[Int] = {
    val u = toEigenspace(adj, constraints)
    (0 until u.length).map(i => if (u.get(i) > 0) 1 else 0)
  }

  def toEigenspace(adj: DoubleMatrix, constraints: DoubleMatrix): DoubleMatrix = {
    val n = adj.columns
    val vol = adj.sum
    val d = degreeMatrix(adj)
    val dNorm = pinv(sqrti(d))
    val lNorm = symLaplacian(adj)
    val qNorm = dNorm.mmul(constraints).mmul(dNorm)
    val maxQLam = symmetricEigenvalues(qNorm).sort().get(n - 1)

    // solve generalized eigenvalue problem
    val q1 = qNorm.sub(eye(n).muli(maxQLam * 0.5))
    val (vectors, values) = generalizedEigenvectors(lNorm, q1)
    // normalize eigenvectors
    for (i <- 0 until vectors.columns) {
      val col = vectors.getColumn(i)
      vectors.putColumn(i, col.divi(euclidNorm(col)).muli(math.pow(vol, 0.5)))
    }

    val I = (0 until n).filter(i => values.get(i) >= 0)
    val feasibleVectors = vectors.getColumns(I.toArray)

    val costs = (0 until feasibleVectors.columns).map { i =>
      val v = feasibleVectors.getColumn(i)
      v.transpose().mmul(lNorm).mmul(v).get(0)
    }
    val ind = costs.zipWithIndex // indices sorted by cost
      .filter(_._1 > 1e-10)
      .minBy(_._1)
      ._2
    dNorm.mmul(feasibleVectors.getColumn(ind))
  }
}
