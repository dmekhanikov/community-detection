package megabyte.communities.experiments.classification

import megabyte.communities.experiments.config.ExperimentConfig.config._
import megabyte.communities.experiments.util.DataUtil._
import megabyte.communities.util.{DataTransformer, IO}
import org.jblas.DoubleMatrix

private class EarlyFusion

object EarlyFusion {

  def main(args: Array[String]): Unit = {
    val networkUsers: Map[String, Seq[Users]] =
      featureFiles.mapValues(_.map(readUsersData)).view.force
    val mergedUsers: Map[String, Users] = mergeUserData(networkUsers)
    val normalizedData: Map[String, Users] = mergedUsers.mapValues(normalizeFeatures)
    val concatUsers: Users = concatFeatures(normalizedData)

    val allLabels: Map[String, String] = readLabels(labelsFile, ID_COL, GENDER_COL)
    val trainIds = IO.readLines(trainIdsFile)
    val testIds = IO.readLines(testIdsFile)

    val trainFeatures = makeFeaturesMatrix(concatUsers, trainIds)
    val testFeatures = makeFeaturesMatrix(concatUsers, testIds)

    val trainLabels = trainIds.map(allLabels)
    val testLabels = testIds.map(allLabels)

    val trainInstances = DataTransformer.constructInstances(trainFeatures, GENDER_VALUES, trainLabels)
    val testInstances = DataTransformer.constructInstances(testFeatures, GENDER_VALUES, testLabels)

    val evaluation = RandomForestClassification.getEvaluation(trainInstances, testInstances)
    RandomForestClassification.printDetailedStats(evaluation)
  }

  private def makeFeaturesMatrix(users: Users, ids: Seq[String]): DoubleMatrix = {
    val features = ids.map(id => users(id).toArray)
    new DoubleMatrix(features.toArray)
  }

  private def concatFeatures(networkUsers: Map[String, Users]): Users = {
    networkUsers.values.reduce { (u1, u2) =>
      val ids = u1.keys
      ids.map { id =>
        id -> (u1(id) ++ u2(id))
      }.toMap
    }
  }
}
