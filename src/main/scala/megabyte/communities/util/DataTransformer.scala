package megabyte.communities.util

import java.util

import org.jblas.DoubleMatrix
import weka.core.{Attribute, DenseInstance, Instances}

object DataTransformer {

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
}
