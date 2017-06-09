package megabyte.communities.experiments.transformer.social

import com.typesafe.scalalogging.Logger
import megabyte.communities.experiments.config.ExperimentConfig.config._
import megabyte.communities.experiments.util.DataUtil._
import megabyte.communities.util.IO

import scala.util.Random

object IdsSelector {

  private val LOG = Logger[IdsSelector.type]

  private val TEST_FRACTION = 0.1

  def main(args: Array[String]): Unit = {
    val networkUsers: Map[String, Seq[Users]] =
      featureFiles.mapValues(_.map(readUsersData))
    LOG.info("Finding users intersection")
    val allLabels: Map[String, String] = readLabels(labelsFile, ID_COL, GENDER_COL)
    val allIds: Seq[String] =
      findUsersIntersection(networkUsers.values.flatten.toSeq)
        .filter(allLabels.contains)
        .toSeq
    val permutation = Random.shuffle[String, Seq](allIds)
    val (testIds, trainIds) = split(permutation, TEST_FRACTION)
    LOG.info("Writing user ids to files")
    IO.writeLines(trainIds, trainIdsFile)
    IO.writeLines(testIds, testIdsFile)
  }
}
