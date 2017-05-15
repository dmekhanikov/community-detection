package megabyte.communities.algo.constraints

import org.jblas.DoubleMatrix

trait ConstraintsApplier {

  def applyConstraints(w: DoubleMatrix, q: DoubleMatrix): DoubleMatrix
}
