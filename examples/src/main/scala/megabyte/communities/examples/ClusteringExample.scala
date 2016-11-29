package megabyte.communities.examples

import megabyte.communities.algo.graph.SpectralClustering
import megabyte.communities.examples.Util._
import megabyte.communities.examples.widget.PointsPane
import megabyte.communities.util.DataTransformer._

object ClusteringExample {
  def main(args: Array[String]) = {
    val pointsMatrix = readPoints("Aggregation.txt")
    val adj = pointsToGraph(pointsMatrix, .5)
    val clustering = SpectralClustering.getClustering(adj, 7)
    val points = (0 until pointsMatrix.rows).map { i =>
      (pointsMatrix.get(i, 0), pointsMatrix.get(i, 1))
    }
    val pointsPane = new PointsPane(points)
    pointsPane.clustering = clustering
  }
}
