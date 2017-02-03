package megabyte.communities.util

import megabyte.communities.util.DoubleMatrixOps._
import org.jblas.DoubleMatrix
import org.scalatest.{FlatSpec, Matchers}

class DoubleMatrixOpsTest extends FlatSpec with Matchers {

  private implicit def intWithTimes(n: Int) = new {
    def times(f: => Unit): Unit = {
      1 to n foreach { _ => f }
    }
  }

  """a \* b""" should "bo equal to a * b when b is diagonal" in {
    50 times {
      val a = genDiagMatrix(15)
      val b = genMatrix(15, 10)
      a \* b should be(a * b)
    }
  }

  it should "not change a nor b" in {
    50 times {
      val a = genDiagMatrix(15)
      val b = genMatrix(15, 10)
      val a1 = a.dup
      val b1 = b.dup
      a \* b
      a should be (a1)
      b should be (b1)
    }
  }

  """a *\= b""" should "be equal to a * b when b is diagonal" in {
    50 times {
      val a = genMatrix(15, 10)
      val b = genDiagMatrix(10)
      val c = a *\= b
      c should be (a * b)
    }
  }

  it should "change a" in {
    50 times {
      val a = genMatrix(15, 10)
      val b = genDiagMatrix(10)
      val a1 = a.dup
      a *\= b
      a should be (a1 * b)
    }
  }

  it should "not change b" in {
    50 times {
      val a = genMatrix(15, 10)
      val b = genDiagMatrix(10)
      val b1 = b.dup
      a *\= b
      b should be (b1)
    }
  }

  private def genMatrix(rows: Int, cols: Int): DoubleMatrix = {
    DoubleMatrix.rand(rows, cols)
  }

  private def genDiagMatrix(n: Int): DoubleMatrix = {
    val diag = new DoubleMatrix(n)
    DoubleMatrix.diag(diag, n, n)
  }
}
