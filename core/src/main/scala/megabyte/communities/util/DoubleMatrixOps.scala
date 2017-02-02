package megabyte.communities.util

import megabyte.communities.util.DoubleMatrixOps._
import org.jblas.DoubleMatrix

final class DoubleMatrixOps(val self: DoubleMatrix) {
  def +(other: DoubleMatrix): DoubleMatrix = self.add(other)

  def +=(other: DoubleMatrix): DoubleMatrix = self.addi(other)

  def -(other: DoubleMatrix): DoubleMatrix = self.sub(other)

  def -=(other: DoubleMatrix): DoubleMatrix = self.subi(other)

  def *(v: Double): DoubleMatrix = self.mul(v)

  def *=(v: Double): DoubleMatrix = self.muli(v)

  def *(other: DoubleMatrix): DoubleMatrix = self.mmul(other)

  def *=(other: DoubleMatrix): DoubleMatrix = self.mmuli(other)

  def /(v: Double): DoubleMatrix = self.div(v)

  def /=(v: Double): DoubleMatrix = self.divi(v)

  def normRowsI(): DoubleMatrix = {
    val n = self.rows
    for (i <- 0 until n) {
      val row = self.getRow(i)
      val norm = row.norm
      for (j <- 0 until self.columns) {
        val x = self.get(i, j)
        self.put(i, j, x / norm)
      }
    }
    self
  }

  def *\=(other: DoubleMatrix): DoubleMatrix = {
    val n = self.rows
    for (i <- 0 until n; j <- 0 until n) {
      val v = self.get(i, j) * other.get(j, j)
      self.put(i, j, v)
    }
    self
  }

  def \*(other: DoubleMatrix): DoubleMatrix = {
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

  def norm: Double = self.norm2

  def distTo(other: DoubleMatrix): Double = {
    (this - other).norm
  }
}

object DoubleMatrixOps {
  implicit def toDoubleMatrixOps(matrix: DoubleMatrix): DoubleMatrixOps = new DoubleMatrixOps(matrix)
}
