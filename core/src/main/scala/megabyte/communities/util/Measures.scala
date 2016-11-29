package megabyte.communities.util

import org.jblas.DoubleMatrix

object Measures {

  def euclidDist(p1: DoubleMatrix, p2: DoubleMatrix): Double = {
    euclidNorm(p1.sub(p2))
  }

  def euclidNorm(p: DoubleMatrix): Double = {
    val n = p.length
    val sqrSum = (0 until n)
      .map { i => math.pow(p.get(i), 2) }
      .sum
    math.sqrt(sqrSum)
  }

  def gaussianSim(p1: DoubleMatrix, p2: DoubleMatrix, sigma: Double): Double = {
    val d = euclidDist(p1, p2)
    Math.exp(-math.pow(d, 2) / 2 / math.pow(sigma, 2))
  }
}
