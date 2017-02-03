package megabyte.communities.algo.points

import megabyte.communities.util.DataTransformer.constructInstances
import org.jblas.DoubleMatrix
import weka.clusterers.{XMeans => WekaXMeans}

object XMeans {

  val MIN_CLUSTERS = 2
  val MAX_CLUSTERS = 100

  def getClustering(points: DoubleMatrix): Array[Int] = {
    val xMeans = new WekaXMeans()
    xMeans.setMinNumClusters(MIN_CLUSTERS)
    xMeans.setMaxNumClusters(MAX_CLUSTERS)
    xMeans.setSeed(System.currentTimeMillis.toInt)
    val instances = constructInstances(points)
    xMeans.buildClusterer(instances)
    (0 until instances.size)
      .map(instances.get)
      .map(xMeans.clusterInstance)
      .toArray
  }
}
