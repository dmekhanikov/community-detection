package megabyte.communities.experiments.classification

import com.typesafe.scalalogging.Logger
import megabyte.communities.experiments.config.ExperimentConfig.config._
import megabyte.communities.experiments.util.DataUtil._
import megabyte.communities.util.IO

object EarlyFusionICA extends ICAPreprocessor {

  private val LOG = Logger[EarlyFusionICA.type]

  def main(args: Array[String]): Unit = {
    val networkUsers: Map[String, Seq[Users]] =
      featureFiles.mapValues(_.map(readUsersData)).view.force
    val mergedUsers: Map[String, Users] = mergeUserData(networkUsers)
    val normalizedData: Map[String, Users] = mergedUsers.mapValues(normalizeUserFeatures)
    val concatUsers: Users = concatFeatures(normalizedData)

    val trainIds = IO.readLines(trainIdsFile)
    val testIds = IO.readLines(testIdsFile)

    val labels: Map[String, String] = readLabels(labelsFile, ID_COL, GENDER_COL)

    val (componentNum, _) = tuneComponentsNum(concatUsers, labels, trainIds)
    val fMeasure = evaluate(concatUsers, labels, trainIds, testIds, componentNum)
    LOG.info(s"Best solution:")
    LOG.info(s"Components: $componentNum; F-Measure: $fMeasure")
  }
}
