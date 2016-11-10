package megabyte.communities.util

import java.util

import org.jblas.DoubleMatrix
import weka.core.{Attribute, DenseInstance, Instances}

object DataTransformer {

  // rows are points
  def constructInstances(points: DoubleMatrix): Instances = {
    val attributes = new util.ArrayList[Attribute](points.rows)
    for (i <- 0 until points.columns) {
      attributes.add(new Attribute("x" + i, i))
    }
    val instances = new Instances("Points", attributes, points.rows)
    for (i <- 0 until points.rows) {
      val inst = new DenseInstance(points.columns)
      for (j <- 0 until points.columns) {
        inst.setValue(j, points.get(i, j))
      }
      instances.add(inst)
    }
    instances
  }

  // points to adjacency matrix of graph with gaussian similarity
  // sigma - parameter for gaussian similarity
  def pointsToGraph(points: DoubleMatrix, sigma: Double): DoubleMatrix = {
    val n = points.rows
    val adj = new DoubleMatrix(n, n)
    for (i <- 0 until n) {
      val p1 = points.getRow(i)
      for (j <- 0 until i) {
        val p2 = points.getRow(j)
        val sim = Measures.gaussianSim(p1, p2, sigma)
        adj.put(i, j, sim)
        adj.put(j, i, sim)
      }
    }
    adj
  }
}
