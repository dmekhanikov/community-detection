package megabyte.communities.experiments.classification

import megabyte.communities.experiments.config.ExperimentConfig.config._
import megabyte.communities.experiments.util.DataUtil._
import megabyte.communities.util.DataTransformer
import org.jblas.DoubleMatrix

import scala.util.Random

private class EarlyFusion

object EarlyFusion {

  private val TEST_FRACTION = 0.1

  def main(args: Array[String]): Unit = {
    val networkUsers: Map[String, Seq[Users]] =
      featureFiles.mapValues(_.map(readUsersData)).view.force
    val mergedUsers: Map[String, Users] = mergeUserData(networkUsers)
    val normalizedData: Map[String, Users] = mergedUsers.mapValues(normalizeFeatures)
    val concatUsers: Users = concatFeatures(normalizedData)

    val allLabels: Map[String, String] = readLabels(labelsFile, ID_COL, GENDER_COL)
    val numeration = concatUsers.keys.filter(allLabels.contains).toSeq
    val permutation = Random.shuffle[Int, Seq](numeration.indices)
    val (testIndices, trainIndices) = split(permutation, TEST_FRACTION)

    val allFeatures = new DoubleMatrix(numeration.map(id => concatUsers(id).toArray).toArray)

    val testLabels = getLabels(testIndices, numeration, allLabels)
    val trainLabels = getLabels(trainIndices, numeration, allLabels)

    val trainFeatures = allFeatures.getRows(trainIndices.toArray)
    val testFeatures = allFeatures.getRows(testIndices.toArray)
    val trainInstances = DataTransformer.constructInstances(trainFeatures, GENDER_VALUES, trainLabels)
    val testInstances = DataTransformer.constructInstances(testFeatures, GENDER_VALUES, testLabels)

    val evaluation = RandomForestClassification.getEvaluation(trainInstances, testInstances)
    RandomForestClassification.printDetailedStats(evaluation)
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
