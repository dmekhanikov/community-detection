package megabyte.communities.algo.points

import org.jblas.DoubleMatrix
import weka.clusterers.SimpleKMeans
import megabyte.communities.util.DataTransformer.constructInstances

object KMeans {

  def getClustering(points: DoubleMatrix, k: Int): Array[Int] = {
    val kMeans = new SimpleKMeans()
    kMeans.setNumClusters(k)
    val instances = constructInstances(points)
    kMeans.setPreserveInstancesOrder(true)
    kMeans.buildClusterer(instances)
    kMeans.getAssignments
  }
}
