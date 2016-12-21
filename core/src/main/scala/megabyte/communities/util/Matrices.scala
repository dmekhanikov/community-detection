package megabyte.communities.util

import org.jblas.DoubleMatrix

object Matrices {

  def diagElements(m: DoubleMatrix): Seq[Double] = {
    (0 until m.columns).map(i => m.get(i, i))
  }

  def nonzeroCount(m: DoubleMatrix): Int = {
    (0 until m.length).count(m.get(_) != 0)
  }

  def normRowsI(matrix: DoubleMatrix): DoubleMatrix = {
    val n = matrix.rows
    for (i <- 0 until n) {
      val row = matrix.getRow(i)
      val norm = Measures.euclidNorm(row)
      for (j <- 0 until row.columns) {
        val x = matrix.get(i, j)
        matrix.put(i, j, x / norm)
      }
    }
    matrix
  }
}
