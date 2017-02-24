package megabyte.communities.examples

import megabyte.communities.algo.graph.{ConstrainedSpectralClustering, SpectralClustering}
import megabyte.communities.examples.widget.PointsPane
import megabyte.communities.util.DataTransformer
import org.jblas.DoubleMatrix

import Util._

object ConstrainedClusteringExample {

  def main(args: Array[String]): Unit = {
    visualize()
  }

  def visualize(): Unit = {
    val pointsMatrix = readPoints("jain.txt")
    val adj = DataTransformer.pointsToGraph(pointsMatrix, 0.8)
    val clustering = SpectralClustering.getClustering(adj, 2)
    val points = (0 until pointsMatrix.rows).map { i =>
      (pointsMatrix.get(i, 0), pointsMatrix.get(i, 1))
    }
    val pointsPane = new PointsPane(points)
    pointsPane.clustering = clustering
    pointsPane.subscribe(() => updateClustering(pointsPane, adj))
  }

  private def updateClustering(pointsPane: PointsPane, adj: DoubleMatrix): Unit = {
    val Q = constraintMatrix(adj.columns, pointsPane.mlConstraints, pointsPane.clConstraints)
    val clustering = ConstrainedSpectralClustering.getClustering(adj, Q, 2)
    pointsPane.clustering = clustering
  }
}
