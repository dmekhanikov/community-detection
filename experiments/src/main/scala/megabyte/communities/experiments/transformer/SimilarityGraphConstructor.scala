package megabyte.communities.experiments.transformer

import java.io.File

import com.typesafe.scalalogging.Logger
import megabyte.communities.experiments.config.ExperimentConfig.config._
import megabyte.communities.experiments.util.DataUtil._
import megabyte.communities.util.DoubleMatrixOps._
import megabyte.communities.util.Measures
import org.jblas.DoubleMatrix

private class SimilarityGraphConstructor

object SimilarityGraphConstructor {

  private val LOG = Logger[SimilarityGraphConstructor]

  private val SIGMA_FACTOR = 1.5

  def main(args: Array[String]): Unit = {
    val networksData: Map[String, Seq[Users]] =
      featureFiles.mapValues(_.map(readUsersData)).view.force
    val mergedData: Map[String, Users] = mergeUserData(networksData)
    val normalizedData: Map[String, Users] = mergedData.mapValues(normalizeFeatures)
    val numeration: Seq[String] = mergedData.values.flatMap(_.keys).toSet.toSeq
    normalizedData.par.foreach { case (net, users) =>
      LOG.info(s"Calculating adjacency matrix for $net")
      val adj = calcAdjMatrix(users, numeration)
      val outFile = new File(similarityGraphsDir, s"$net.csv")
      LOG.info(s"Writing result for $net to $outFile")
      adj.write(outFile, header = Some(numeration))
    }
    LOG.info("Finished, exiting")
  }

  private def calcAdjMatrix(users: Users, numeration: Seq[String]): DoubleMatrix = {
    val n = users.keys.size
    val distances = new DoubleMatrix(n, n)
    for (i <- (0 until n).par; j <- (i + 1 until n).par) {
      val a = users(numeration(i))
      val b = users(numeration(j))
      val diff = a.zip(b).map { case (x, y) => x - y }
      val dist = Measures.euclidNorm(diff)
      distances.put(i, j, dist)
      distances.put(j, i, dist)
    }
    val sigma = SIGMA_FACTOR * distances.data.sorted.apply(n * n / 2)
    val adj = distances.map { dist =>
      math.exp(-math.pow(dist, 2) / 2 / math.pow(sigma, 2))
    }
    0 until n foreach (i => adj.put(i, i, 0))
    adj
  }
}
