package megabyte.communities.experiments.classification.social

import com.typesafe.scalalogging.Logger
import megabyte.communities.experiments.classification.Evaluator
import megabyte.communities.experiments.config.ExperimentConfig.config._
import megabyte.communities.experiments.util.DataUtil._
import megabyte.communities.util.{DataTransformer, IO}
import weka.classifiers.trees.RandomForest

object SingleLayer {

  private val LOG = Logger[SingleLayer.type]

  def main(args: Array[String]): Unit = {
    val networkUsers: Map[String, Seq[Users]] =
      featureFiles.mapValues(_.map(readUsersData)).view.force
    val mergedUsers: Map[String, Users] = mergeUserData(networkUsers)
    val normalizedData: Map[String, Users] = mergedUsers.mapValues(normalizeUserFeatures)

    val trainIds = IO.readLines(trainIdsFile)
    val testIds = IO.readLines(testIdsFile)

    val allLabels: Map[String, String] = readLabels(groundFile, ID_COL, GENDER_COL)
    val trainLabels = trainIds.map(allLabels)
    val testLabels = testIds.map(allLabels)

    val randomForest = new RandomForest
    for ((net, users) <- normalizedData) {
      LOG.info(s"Processing $net")
      val trainFeatures = makeFeaturesMatrix(users, trainIds)
      val testFeatures = makeFeaturesMatrix(users, testIds)

      val trainInstances = DataTransformer.constructInstances(trainFeatures, GENDER_VALUES, trainLabels)
      val testInstances = DataTransformer.constructInstances(testFeatures, GENDER_VALUES, testLabels)

      val evaluation = Evaluator.getEvaluation(randomForest, trainInstances, testInstances)
      LOG.info(s"Results for $net:")
      Evaluator.printDetailedStats(evaluation)
    }
  }
}
