package megabyte.communities.experiments.classification.social

import com.typesafe.scalalogging.Logger
import megabyte.communities.experiments.classification.Evaluator
import megabyte.communities.experiments.config.ExperimentConfig.config._
import megabyte.communities.experiments.util.DataUtil._
import megabyte.communities.util.{DataTransformer, IO}
import weka.classifiers.trees.RandomForest

object SingleLayerAutoEncoder extends AutoEncoderPreprocessor {

  private val LOG = Logger[SingleLayerAutoEncoder.type]

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

    val twitterUsers = normalizedData("twitter")
    val transformedUsers = transform(twitterUsers)

    val trainFeaturesSeq = trainIds.map(transformedUsers)
    val testFeaturesSeq = testIds.map(transformedUsers)

    val trainInstances = DataTransformer.constructInstances(trainFeaturesSeq, GENDER_VALUES, trainLabels)
    val testInstances = DataTransformer.constructInstances(testFeaturesSeq, GENDER_VALUES, testLabels)

    val evaluation = Evaluator.getEvaluation(new RandomForest, trainInstances, testInstances)
    Evaluator.printDetailedStats(evaluation)
  }
}
