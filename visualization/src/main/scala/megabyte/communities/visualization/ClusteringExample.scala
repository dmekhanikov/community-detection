package megabyte.communities.visualization

import megabyte.communities.algo.graph.SpectralClustering
import megabyte.communities.visualization.Util._
import megabyte.communities.visualization.widget.PointsPane
import megabyte.communities.util.DataTransformer._

object ClusteringExample {

  private val file = "Aggregation.txt"
  private val k = 7
  private val sigma = 0.5

  def main(args: Array[String]) = {
    val pointsMatrix = readPoints(file)
    val adj = pointsToGraph(pointsMatrix, sigma)
    val clustering = SpectralClustering.getClustering(adj, k)
    val points = (0 until pointsMatrix.rows).map { i =>
      (pointsMatrix.get(i, 0), pointsMatrix.get(i, 1))
    }
    val pointsPane = new PointsPane(points)
    pointsPane.clustering = clustering
  }
}
