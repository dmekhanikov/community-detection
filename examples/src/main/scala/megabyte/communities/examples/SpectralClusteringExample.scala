package megabyte.communities.examples

import java.io.FileNotFoundException

import megabyte.communities.algo.graph.{ConstrainedSpectralClustering, SpectralClustering}
import megabyte.communities.examples.widget.PointsPane
import megabyte.communities.util.DataTransformer
import org.jblas.DoubleMatrix

import scala.io.Source

object SpectralClusteringExample {

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

  def updateClustering(pointsPane: PointsPane, adj: DoubleMatrix): Unit = {
    val Q = constraintMatrix(adj.columns, pointsPane.mlConstraints, pointsPane.clConstraints)
    val clustering = ConstrainedSpectralClustering.getClustering(adj, Q)
    pointsPane.clustering = clustering
  }

  def constraintMatrix(n: Int, mlConstraints: Seq[(Int, Int)], clConstraints: Seq[(Int, Int)]): DoubleMatrix = {
    val Q = new DoubleMatrix(n, n)
    mlConstraints.foreach { case (i, j) =>
        Q.put(i, j, 1)
        Q.put(j, i, 1)
    }
    clConstraints.foreach { case (i, j) =>
      Q.put(i, j, -1)
      Q.put(j, i, -1)
    }
    Q
  }

  def readPoints(resource: String): DoubleMatrix = {
    val instances = Option(getClass.getClassLoader.getResourceAsStream(resource))
      .map(Source.fromInputStream(_).getLines)
      .getOrElse(throw new FileNotFoundException(resource))
      .map(_.split("\\s").map(_.toDouble))
      .toSeq
    val n = instances.size
    val m = instances.head.length - 1
    val matrix = new DoubleMatrix(n, m)
    for (i <- 0 until n) {
      for (j <- 0 until m) {
        matrix.put(i, j, instances(i)(j))
      }
    }
    matrix
  }
}
