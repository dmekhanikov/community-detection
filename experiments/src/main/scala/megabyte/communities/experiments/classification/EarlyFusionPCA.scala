package megabyte.communities.experiments.classification

import java.io.File

import com.typesafe.scalalogging.Logger
import megabyte.communities.experiments.config.ExperimentConfig.config._
import megabyte.communities.experiments.util.DataUtil._
import megabyte.communities.util.{DataTransformer, IO}

private class EarlyFusionPCA

object EarlyFusionPCA extends PCAPreprocessor {

  private val LOG = Logger[EarlyFusionPCA]

  private val relationFile = new File(relationsDir, "multilayer-pca.csv")

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

    val relation = getRelation(trainInstances, testInstances)
    val (k, fMeasure) = relation.maxBy(_._2)
    LOG.info(s"Best solution: F-measure=$fMeasure (k=$k)")
    IO.writeRelation(Seq("k", "F-measure"), relation, relationFile)
  }
}
