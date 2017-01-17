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

  implicit def richDoubleMatrix(self: DoubleMatrix) = new {

    def mulByDiagI(other: DoubleMatrix): DoubleMatrix = {
      val n = self.rows
      for (i <- 0 until n; j <- 0 until n) {
        val v = self.get(i, j) * other.get(j, j)
        self.put(i, j, v)
      }
      self
    }

    def mulDiag(other: DoubleMatrix): DoubleMatrix = {
      val n = self.rows
      val result = new DoubleMatrix(n, n)
      for (i <- 0 until n; j <- 0 until n) {
        val v = other.get(i, j) * self.get(i, i)
        result.put(i, j, v)
      }
      self
    }

    def invDiagI(): DoubleMatrix = {
      for (i <- 0 until self.rows) {
        val v = self.get(i, i)
        val inv = v match {
          case 0 => 0
          case _ => 1 / v
        }
        self.put(i, i, inv)
      }
      self
    }

    def sqrtDiagI(): DoubleMatrix = {
      for (i <- 0 until self.rows) {
        val v = self.get(i, i)
        self.put(i, i, Math.sqrt(v))
      }
      self
    }
  }
}
