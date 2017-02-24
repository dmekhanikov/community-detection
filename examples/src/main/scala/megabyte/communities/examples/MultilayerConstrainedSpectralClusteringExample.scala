package megabyte.communities.examples

import megabyte.communities.algo.graph.{MultilayerConstrainedSpectralClustering, MultilayerSpectralClustering}
import megabyte.communities.examples.Util._
import megabyte.communities.examples.widget.PointsPane
import megabyte.communities.util.DataTransformer._
import org.jblas.DoubleMatrix

import scala.util.Random

object MultilayerConstrainedSpectralClusteringExample {

  private val random = new Random(System.currentTimeMillis)

  private val file = "jain.txt"
  private val k = 2
  private val layers = 5
  private val alpha = 0.1
  private val sigma = 0.8
  private val p = 0.1

  def main(args: Array[String]): Unit = {
    val pointsMatrix = readPoints(file)
    val adjs = stratify(pointsToGraph(pointsMatrix, sigma), layers, p)
    val clustering = MultilayerSpectralClustering.getClustering(adjs, k, alpha)
    val points = (0 until pointsMatrix.rows).map { i =>
      (pointsMatrix.get(i, 0), pointsMatrix.get(i, 1))
    }
    val pointsPane = new PointsPane(points)
    pointsPane.clustering = clustering
    pointsPane.subscribe(() => updateClustering(pointsPane, adjs))
  }

  private def updateClustering(pointsPane: PointsPane, adjs: Seq[DoubleMatrix]): Unit = {
    val n = adjs.head.columns
    val m = adjs.size
    val Q = constraintMatrix(n, pointsPane.mlConstraints, pointsPane.clConstraints)
    val clustering = MultilayerConstrainedSpectralClustering.getClustering(adjs, List.fill(m)(Q), k, alpha)
    pointsPane.clustering = clustering
  }

  // layers - how many layers you need
  // p - what fraction of edges you want to leave
  private def stratify(adj: DoubleMatrix, layers: Int, p: Double): Seq[DoubleMatrix] = {
    for (i <- 1 to layers) yield dropEdges(adj.dup(), p)
  }

  private def dropEdges(adj: DoubleMatrix, p: Double): DoubleMatrix = {
    val n = adj.columns
    val pairs = (0 until n).flatMap { i =>
      for (j <- i + 1 until n; if j != i) yield (i, j)
    }
    val zeroCount = (pairs.length * (1 - p)).toInt
    random.shuffle(pairs).slice(0, zeroCount).foreach { case (i, j) =>
      adj.put(i, j, 0)
      adj.put(j, i, 0)
    }
    adj
  }
}
