package megabyte.communities.experiments.transformer

import java.io.File

import com.typesafe.scalalogging.Logger
import megabyte.communities.experiments.config.ExperimentConfig.config._
import megabyte.communities.experiments.util.DataUtil._
import megabyte.communities.util.DataTransformer
import megabyte.communities.util.DoubleMatrixOps._

object SimilarityGraphConstructor {

  private val LOG = Logger[SimilarityGraphConstructor.type]

  val SIGMA_FACTOR = 1.5

  def main(args: Array[String]): Unit = {
    val networksData: Map[String, Seq[Users]] =
      featureFiles.mapValues(_.map(readUsersData)).view.force
    val mergedData: Map[String, Users] = mergeUserData(networksData)
    val normalizedData: Map[String, Users] = mergedData.mapValues(normalizeFeatures)
    val numeration: Seq[String] = mergedData.values.flatMap(_.keys).toSet.toSeq
    normalizedData.par.foreach { case (net, users) =>
      LOG.info(s"Calculating adjacency matrix for $net")
      val objects = numeration.map { id => users(id) }
      val adj = DataTransformer.heatWeightMatrix(objects, SIGMA_FACTOR)
      val outFile = new File(similarityGraphsDir, s"$net.csv")
      LOG.info(s"Writing result for $net to $outFile")
      adj.write(outFile, header = Some(numeration))
    }
    LOG.info("Finished, exiting")
  }
}
