package megabyte.communities.examples

import megabyte.communities.algo.graph.{ConstrainedSpectralClustering, SpectralClustering}
import megabyte.communities.examples.Util._
import megabyte.communities.examples.widget.PointsPane
import megabyte.communities.util.DataTransformer
import org.jblas.DoubleMatrix

object ConstrainedClusteringExample {

  private val file = "jain.txt"
  private val k = 2
  private val sigma = 0.8

  def main(args: Array[String]): Unit = {
    val pointsMatrix = readPoints(file)
    val adj = DataTransformer.pointsToGraph(pointsMatrix, sigma)
    val clustering = SpectralClustering.getClustering(adj, k)
    val points = (0 until pointsMatrix.rows).map { i =>
      (pointsMatrix.get(i, 0), pointsMatrix.get(i, 1))
    }
    val pointsPane = new PointsPane(points)
    pointsPane.clustering = clustering
    pointsPane.subscribe(() => updateClustering(pointsPane, adj))
  }

  private def updateClustering(pointsPane: PointsPane, adj: DoubleMatrix): Unit = {
    val Q = constraintMatrix(adj.columns, pointsPane.mlConstraints, pointsPane.clConstraints)
    val clustering = ConstrainedSpectralClustering.getClustering(adj, Q, k)
    pointsPane.clustering = clustering
  }
}
