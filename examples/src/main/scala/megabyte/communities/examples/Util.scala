package megabyte.communities.examples

import java.io.FileNotFoundException

import org.jblas.DoubleMatrix

import scala.io.Source

object Util {

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
}
