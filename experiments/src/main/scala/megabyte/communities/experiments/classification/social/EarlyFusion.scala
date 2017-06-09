package megabyte.communities.experiments.classification.social

import megabyte.communities.experiments.classification.Evaluator
import megabyte.communities.experiments.config.ExperimentConfig.config._
import megabyte.communities.experiments.util.DataUtil._
import megabyte.communities.util.{DataTransformer, IO}
import weka.classifiers.trees.RandomForest

object EarlyFusion {

  def main(args: Array[String]): Unit = {
    val networkUsers: Map[String, Seq[Users]] =
      featureFiles.mapValues(_.map(readUsersData)).view.force
    val mergedUsers: Map[String, Users] = mergeUserData(networkUsers)
    val normalizedData: Map[String, Users] = mergedUsers.mapValues(normalizeUserFeatures)
    val concatUsers: Users = concatFeatures(normalizedData)

    val allLabels: Map[String, String] = readLabels(groundFile, ID_COL, GENDER_COL)
    val trainIds = IO.readLines(trainIdsFile)
    val testIds = IO.readLines(testIdsFile)

    val trainFeatures = makeFeaturesMatrix(concatUsers, trainIds)
    val testFeatures = makeFeaturesMatrix(concatUsers, testIds)

    val trainLabels = trainIds.map(allLabels)
    val testLabels = testIds.map(allLabels)

    val trainInstances = DataTransformer.constructInstances(trainFeatures, GENDER_VALUES, trainLabels)
    val testInstances = DataTransformer.constructInstances(testFeatures, GENDER_VALUES, testLabels)

    val randomForest = new RandomForest
    val evaluation = Evaluator.getEvaluation(randomForest, trainInstances, testInstances)
    Evaluator.printDetailedStats(evaluation)
  }
}
