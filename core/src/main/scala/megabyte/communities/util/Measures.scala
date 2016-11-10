package megabyte.communities.util

import org.jblas.DoubleMatrix

object Measures {

  def euclid(p1: DoubleMatrix, p2: DoubleMatrix): Double = {
    val n = p1.columns
    val sqrSum = (0 until n)
      .map { i => math.pow(p1.get(i) - p2.get(i), 2) }
      .sum
    math.sqrt(sqrSum)
  }

  def gaussianSim(p1: DoubleMatrix, p2: DoubleMatrix, sigma: Double): Double = {
    val d = euclid(p1, p2)
    Math.exp(-math.pow(d, 2) / 2 / math.pow(sigma, 2))
  }
}
