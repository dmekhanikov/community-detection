package megabyte.communities.util

import java.util

import megabyte.communities.util.DoubleMatrixOps._
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

  def constructInstances(allFeatures: DoubleMatrix, k: Int, indices: Seq[Int],
                         labelValues: Seq[String], labels: Seq[String]): Instances = {
    val features = allFeatures.prefixColumns(k).getRows(indices.toArray)
    DataTransformer.constructInstances(features, labelValues, labels)
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
    similarityMatrix(points) { (p1, p2) =>
      Measures.gaussianSim(p1, p2, sigma)
    }
  }

  def similarityMatrix(points: DoubleMatrix)(sim: (DoubleMatrix, DoubleMatrix) => Double): DoubleMatrix = {
    val n = points.rows
    val m = new DoubleMatrix(n, n)
    for (i <- 0 until n) {
      val p1 = points.getRow(i)
      for (j <- 0 to i) {
        val p2 = points.getRow(j)
        val v = sim(p1, p2)
        m.put(i, j, v)
        m.put(j, i, v)
      }
    }
    m
  }

  def heatWeightMatrix(objects: Seq[Seq[Double]], sigmaFactor: Double): DoubleMatrix = {
    val n = objects.size
    val distances = new DoubleMatrix(n, n)
    for (i <- (0 until n).par; j <- (i + 1 until n).par) {
      val a = objects(i)
      val b = objects(j)
      val diff = a.zip(b).map { case (x, y) => x - y }
      val dist = Measures.euclidNorm(diff)
      distances.put(i, j, dist)
      distances.put(j, i, dist)
    }
    val sigma = sigmaFactor * distances.data.sorted.apply(n * n / 2)
    val adj = distances.map { dist =>
      math.exp(-math.pow(dist, 2) / 2 / math.pow(sigma, 2))
    }
    0 until n foreach (i => adj.put(i, i, 0))
    adj
  }

  def foldFeatures(values: Iterable[Seq[Double]], f: (Double, Double) => Double): Seq[Double] = {
    values.tail.fold(values.head) { (min: Seq[Double], cur: Seq[Double]) =>
      min.zip(cur).map { case (a, b) => f(a, b) }
    }
  }

  def normalizeFeatures(objects: Iterable[Seq[Double]]): Iterable[Seq[Double]] = {
    val mins = foldFeatures(objects, math.min)
    val maxs = foldFeatures(objects, math.max)
    objects.map { features =>
      features.zip(mins.zip(maxs))
        .map { case (value, (min, max)) =>
          if (min != max) {
            (value - min) / (max - min)
          } else {
            value
          }
        }
    }
  }
}
