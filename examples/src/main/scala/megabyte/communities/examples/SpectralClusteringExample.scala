package megabyte.communities.examples

import java.io.FileNotFoundException

import megabyte.communities.algo.graph.SpectralClustering
import megabyte.communities.util.DataTransformer
import org.jblas.DoubleMatrix

import scala.io.Source

object SpectralClusteringExample {

  def main(args: Array[String]): Unit = {
    val points = readPoints("jain.txt")
    val adj = DataTransformer.pointsToGraph(points, 1)
    val clustering = SpectralClustering.getClustering(adj, 2)
    clustering.zipWithIndex.foreach { case (cluster, i) =>
      println((i + 1) + ": " + cluster)
    }
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
