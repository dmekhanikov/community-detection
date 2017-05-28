package megabyte.communities.visualization

import megabyte.communities.algo.constraints.CustomConstraintsApplier
import megabyte.communities.algo.graph.{ConstrainedSpectralClustering, SpectralClustering}
import megabyte.communities.algo.points.XMeans
import megabyte.communities.util.DataTransformer.pointsToGraph
import megabyte.communities.util.DoubleMatrixOps._
import megabyte.communities.util.Graphs.symLaplacian
import megabyte.communities.visualization.Util.constraintMatrix
import megabyte.communities.visualization.widget.PointsPane
import org.jblas.DoubleMatrix

import scala.util.Random

object TransformationExample {

  private val n1 = 50
  private val n2 = 50
  private val n3 = 50
  private val n = n1 + n2 + n3

  private val bl1 = (0.0, 0.0)
  private val tr1 = (1.0, 1.0)

  private val bl2 = (0.0, 2.0)
  private val tr2 = (1.0, 3.0)

  private val bl3 = (2.0, 1.0)
  private val tr3 = (3.0, 2.0)

  private val random = new Random(1)

  private val sigma = 0.5
  private val knn = 30

  private val points = fillUniform(bl1, tr1, n1) ++ fillUniform(bl2, tr2, n2) ++ fillUniform(bl3, tr3, n3)
  private val pointsMatrix = pointsToMatrix(points)
  private val w = pointsToGraph(pointsMatrix, sigma)

  def main(args: Array[String]): Unit = {
    val lSym = symLaplacian(w)
    val subspace = SpectralClustering.toEigenspace(lSym).sliceColumns(1, 3)
    val newPoints = matrixToPoints(subspace)
    val pointsPane = new PointsPane(newPoints)
    pointsPane.clustering = XMeans.getClustering(subspace)
    pointsPane.subscribe(() => updateClustering(pointsPane, pointsMatrix))
  }

  private def updateClustering(pointsPane: PointsPane, points: DoubleMatrix): Unit = {
    val q = constraintMatrix(n, pointsPane.mlConstraints, pointsPane.clConstraints)
    val constraintsApplier = new CustomConstraintsApplier(knn)
    val constrainedClustering = new ConstrainedSpectralClustering(constraintsApplier)
    val subspace = constrainedClustering.toEigenspace(w, q).sliceColumns(1, 3)
    pointsPane.setPoints(matrixToPoints(subspace))
    pointsPane.clustering = XMeans.getClustering(subspace)
  }

  private def fillUniform(bl: (Double, Double), tr: (Double, Double), n: Int): Seq[(Double, Double)] = {
    for (_ <- 0 until n) yield {
      val x = random.nextDouble * (tr._1 - bl._1) + bl._1
      val y = random.nextDouble * (tr._2 - bl._2) + bl._2
      (x, y)
    }
  }

  private def pointsToMatrix(points: Seq[(Double, Double)]): DoubleMatrix = {
    val matrix = new DoubleMatrix(points.size, 2)
    points.zipWithIndex.map { case (p, i) =>
        matrix.put(i, 0, p._1)
        matrix.put(i, 1, p._2)
    }
    matrix
  }

  private def matrixToPoints(pointsMatrix: DoubleMatrix): Seq[(Double, Double)] = {
    for (i <- 0 until pointsMatrix.rows) yield {
      (pointsMatrix.get(i, 0), pointsMatrix.get(i, 1))
    }
  }
}
