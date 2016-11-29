package megabyte.communities.util

import com.github.fommil.netlib.F2jLAPACK
import org.jblas.DoubleMatrix
import org.jblas.exceptions.LapackException
import org.netlib.util.intW

object Eigen {

  private val lapack = new F2jLAPACK()
  private val dummy = new Array[Double](0)

  def generalizedEigenvectors(a: DoubleMatrix, b: DoubleMatrix): (DoubleMatrix, DoubleMatrix) = {
    val n = a.rows
    val da = a.dup()
    val db = b.dup()
    val alphar = new Array[Double](n)
    val alphai = new Array[Double](n)
    val beta = new Array[Double](n)
    val vr = new Array[Double](n * n)
    val info = new intW(0)
    var work = new Array[Double](1)
    lapack.dggev("N", "V", n, dummy, da.rows, dummy, db.rows, dummy, dummy, dummy, dummy, 1, dummy, n, work, -1, info)
    if (info.`val` != 0) {
      throw new LapackException("Cannot compute optimal lwork")
    }
    val lwork = work(0).toInt
    work = new Array[Double](lwork)
    lapack.dggev("N", "V", n, da.data, da.rows, db.data, db.rows, alphar, alphai, beta, dummy, 1, vr, n, work, lwork, info)
    if (info.`val` != 0) {
      throw new LapackException("Cannot compute generalized eigenvectors")
    }

    (new DoubleMatrix(n, n, vr:_*), new DoubleMatrix(alphar))
  }
}
