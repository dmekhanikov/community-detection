package megabyte.communities.experiments.classification.social

import com.typesafe.scalalogging.Logger
import megabyte.communities.experiments.config.ExperimentConfig.config._
import megabyte.communities.experiments.util.DataUtil._
import megabyte.communities.util.IO

object SingleLayerICA extends ICAPreprocessor {

  private val LOG = Logger[SingleLayerICA.type]

  def main(args: Array[String]): Unit = {
    val networkUsers: Map[String, Seq[Users]] =
      featureFiles.mapValues(_.map(readUsersData)).view.force
    val mergedUsers: Map[String, Users] = mergeUserData(networkUsers)
    val normalizedData: Map[String, Users] = mergedUsers.mapValues(normalizeUserFeatures)

    val trainIds = IO.readLines(trainIdsFile)
    val testIds = IO.readLines(testIdsFile)

    val labels: Map[String, String] = readLabels(groundFile, ID_COL, GENDER_COL)

    for ((net, users) <- normalizedData) {
      LOG.info("Evaluating " + net)
      val (componentNum, _) = tuneComponentsNum(users, labels, trainIds)
      val fMeasure = evaluate(users, labels, trainIds, testIds, componentNum)
      LOG.info(s"Best solution for $net:")
      LOG.info(s"Components: $componentNum; F-Measure: $fMeasure")
    }
  }
}
