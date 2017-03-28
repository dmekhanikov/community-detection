package megabyte.communities.util

import java.util

import org.jblas.DoubleMatrix
import weka.core.{Attribute, DenseInstance, Instances}

import scala.collection.JavaConversions._

object DataTransformer {

  // rows are points
  def constructInstances(points: DoubleMatrix): Instances = {
    val attributes = makeAttributes(points.columns)
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

  def constructInstances(features: DoubleMatrix, labelValues: Seq[String], labels: Seq[String]): Instances = {
    val attributes = makeAttributes(features.columns)
    val labelAttr = new Attribute("label", labelValues, features.columns)
    attributes.add(labelAttr)
    val instances = new Instances("Instances", attributes, labels.size)
    for (i <- 0 until features.rows) {
      val inst = new DenseInstance(features.columns + 1)
      inst.setDataset(instances)
      for (j <- 0 until features.columns) {
        inst.setValue(j, features.get(i, j))
      }
      inst.setValue(labelAttr, labels(i))
      instances.add(inst)
    }
    instances.setClass(labelAttr)
    instances
  }

  private def makeAttributes(n: Int): util.ArrayList[Attribute] = {
    val attributes = new util.ArrayList[Attribute](n)
    for (i <- 0 until n) {
      attributes.add(new Attribute("x" + i, i))
    }
    attributes
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
