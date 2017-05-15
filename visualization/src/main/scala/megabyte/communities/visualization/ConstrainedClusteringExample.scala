package megabyte.communities.visualization

import megabyte.communities.algo.constraints.LuConstraintsApplier
import megabyte.communities.algo.graph.{ConstrainedSpectralClustering, SpectralClustering}
import megabyte.communities.util.DataTransformer
import megabyte.communities.visualization.Util._
import megabyte.communities.visualization.widget.PointsPane
import org.jblas.DoubleMatrix

object ConstrainedClusteringExample {

  private val file = "jain.txt"
  private val k = 2
  private val sigma = 0.8
  private val alpha = 0.4

  def main(args: Array[String]): Unit = {
    val pointsMatrix = readPoints(file)
    val adj = DataTransformer.pointsToGraph(pointsMatrix, sigma)
    val clustering = SpectralClustering.getClustering(adj, k)
    val points = (0 until pointsMatrix.rows).map { i =>
      (pointsMatrix.get(i, 0), pointsMatrix.get(i, 1))
    }
    val pointsPane = new PointsPane(points)
    pointsPane.clustering = clustering
    pointsPane.subscribe(() => updateClustering(pointsPane, pointsMatrix))
  }

  private def updateClustering(pointsPane: PointsPane, points: DoubleMatrix): Unit = {
    val n = points.rows
    val w = DataTransformer.pointsToGraph(points, sigma)
    val q = constraintMatrix(n, pointsPane.mlConstraints, pointsPane.clConstraints)
    val constraintsApplier = new LuConstraintsApplier(alpha)
    val constrainedClustering = new ConstrainedSpectralClustering(constraintsApplier)
    pointsPane.clustering = constrainedClustering.getClustering(w, q, k)
  }
}
