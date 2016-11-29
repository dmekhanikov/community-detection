package megabyte.communities.util

import org.jblas.DoubleMatrix

object Matrices {

  def diagElements(m: DoubleMatrix): Seq[Double] = {
    (0 until m.columns).map(i => m.get(i, i))
  }

  def nonzeroCount(m: DoubleMatrix): Int = {
    (0 until m.length).count(m.get(_) != 0)
  }
}
