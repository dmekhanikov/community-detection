package megabyte.communities.algo.constraints

import megabyte.communities.util.DoubleMatrixOps._
import org.jblas.DoubleMatrix

// works only for must-link constraints
class CustomConstraintsApplier(private val knn: Int) extends ConstraintsApplier {

  override def applyConstraints(w: DoubleMatrix, q: DoubleMatrix): DoubleMatrix = {
    val n = w.columns
    constructClosure(q)
    val neighborsMapping: Seq[Seq[Int]] = getNeighborsMapping(w, knn)
    val wMod = new DoubleMatrix(n, n)
      .transform { (i, j, _) =>
        if (q.get(i, j) > 0) {
          1
        } else {
          w.get(i, j)
        }
      }
    for (i <- 0 until n; j <- 0 until n) {
      if (q.get(i, j) > 0) {
        for (k <- neighborsMapping(i); l <- neighborsMapping(j)) {
          val newW = math.max(wMod.get(k, l), wMod.get(k, i) * wMod.get(i, j) * wMod.get(j, l))
          wMod.put(k, l, newW)
        }
      }
    }
    wMod
  }

  private def constructClosure(m: DoubleMatrix): Unit = {
    val n = m.columns
    for (i <- 0 until n; j <- 0 until i; k <- 0 until n) { // i -> k -> j => (i -> j) + (j -> i)
      if (m.get(i, k) > 0 && m.get(k, j) > 0) {
        m.put(i, j, 1)
        m.put(j, i, 1)
      }
    }
  }

  private def getNeighborsMapping(w: DoubleMatrix, knn: Int): Seq[Seq[Int]] = {
    val n = w.columns
    for (i <- 0 until n) yield {
      (0 until n)
        .map { j =>
          (j, w.get(i, j))
        }
        .sortBy(-_._2)
        .map(_._1)
        .take(knn)
    }
  }
}