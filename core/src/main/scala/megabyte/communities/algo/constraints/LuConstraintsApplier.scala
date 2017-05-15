package megabyte.communities.algo.constraints
import megabyte.communities.util.DoubleMatrixOps._
import megabyte.communities.util.Graphs.degreeMatrix
import org.jblas.DoubleMatrix
import org.jblas.DoubleMatrix.eye

class LuConstraintsApplier(private val alpha: Double) extends ConstraintsApplier {

  override def applyConstraints(w: DoubleMatrix, q: DoubleMatrix): DoubleMatrix = {
    val f = propagateConstraints(q, w)
    modifyAdj(w, f)
  }

  def propagateConstraints(q: DoubleMatrix, w: DoubleMatrix): DoubleMatrix = {
    val dNorm = degreeMatrix(w).sqrtDiagI().invDiagI()
    val l = (dNorm * w) *= dNorm
    val t = (eye(w.columns) -= (l *= alpha)).inv()
    val f = ((t * q) *= t) *= math.pow(1 - alpha, 2)
    f
  }

  def modifyAdj(w: DoubleMatrix, q: DoubleMatrix): DoubleMatrix = {
    val n = w.columns
    val adjMod = new DoubleMatrix(n, n)
    for (i <- 0 until n; j <- 0 until n) {
      val f = q.get(i, j)
      val oldVal = w.get(i, j)
      val newVal = if (f >= 0) {
        1 - (1 - f) * (1 - oldVal)
      } else {
        (1 + f) * oldVal
      }
      adjMod.put(i, j, newVal)
    }
    adjMod
  }

  private def makeKNNWeightMatrix(w: DoubleMatrix, knn: Int): DoubleMatrix = {
    val wKnn = new DoubleMatrix(w.rows, w.columns)
    for (i <- 0 until w.rows) {
      val connections = (0 until w.columns)
        .filter(j => j != i)
        .map(j => (j, w.get(i, j)))
      val neighbours = connections
        .sortBy(-_._2)
        .take(knn)
        .map(_._1)
      for (j <- neighbours) {
        val x = w.get(i, j) / sqrtOrOne(w.get(i, i)) / sqrtOrOne(w.get(j, j))
        wKnn.put(i, j, x)
      }
    }
    (wKnn += wKnn.transpose()) /= 2
  }

  private def sqrtOrOne(x: Double): Double = {
    if (x > 0) {
      math.sqrt(x)
    } else {
      1
    }
  }
}
