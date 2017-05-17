package megabyte.communities.algo.constraints

import com.typesafe.scalalogging.Logger
import megabyte.communities.algo.constraints.CustomConstraintsApplier.LOG
import megabyte.communities.util.DoubleMatrixOps._
import org.jblas.DoubleMatrix

object CustomConstraintsApplier {

  private val LOG = Logger[CustomConstraintsApplier.type]

  def constructClosure(m: DoubleMatrix): Unit = {
    LOG.debug("constructing closure for constraints set")
    val n = m.columns
    for (i <- (0 until n).par; j <- 0 until i; k <- 0 until n) { // i -> k -> j => (i -> j) + (j -> i)
      if (m.get(i, k) > 0 && m.get(k, j) > 0) {
        val newVal = math.max(m.get(i, j), m.get(i, k) * m.get(k, j))
        m.put(i, j, newVal)
        m.put(j, i, newVal)
      }
    }
  }
}

// works only for must-link constraints
class CustomConstraintsApplier(private val knn: Int) extends ConstraintsApplier {

  override def applyConstraints(w: DoubleMatrix, q: DoubleMatrix): DoubleMatrix = {
    val n = w.columns
    val neighborsMapping: Seq[Seq[Int]] = getNeighborsMapping(w, knn)
    LOG.debug("imposing constraints")
    val wMod = new DoubleMatrix(n, n)
      .transform { (i, j, _) =>
        math.max(q.get(i, j), w.get(i, j))
      }
    LOG.debug("modifying graph")
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

  private def getNeighborsMapping(w: DoubleMatrix, knn: Int): Seq[Seq[Int]] = {
    LOG.debug("constructing neighbors mapping")
    val n = w.columns
    (for (i <- (0 until n).par) yield {
      (0 until n)
        .map { j =>
          (j, w.get(i, j))
        }
        .sortBy(-_._2)
        .map(_._1)
        .take(knn)
    }).seq
  }
}
