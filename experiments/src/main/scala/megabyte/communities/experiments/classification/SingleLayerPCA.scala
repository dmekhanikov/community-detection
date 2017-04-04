package megabyte.communities.experiments.classification

import java.io.File

import com.typesafe.scalalogging.Logger
import megabyte.communities.experiments.config.ExperimentConfig.config._
import megabyte.communities.experiments.util.DataUtil._
import megabyte.communities.util.{DataTransformer, IO}

object SingleLayerPCA extends PCAPreprocessor {

  private val LOG = Logger[SingleLayerPCA.type]

  private val singleLayerPCARelationsDir = new File(relationsDir, "single_layer_pca")

  def main(args: Array[String]): Unit = {
    val networkUsers: Map[String, Seq[Users]] =
      featureFiles.mapValues(_.map(readUsersData)).view.force
    val mergedUsers: Map[String, Users] = mergeUserData(networkUsers)
    val normalizedData: Map[String, Users] = mergedUsers.mapValues(normalizeFeatures)

    val allLabels: Map[String, String] = readLabels(labelsFile, ID_COL, GENDER_COL)
    val trainIds = IO.readLines(trainIdsFile)
    val testIds = IO.readLines(testIdsFile)

    val trainLabels = trainIds.map(allLabels)
    val testLabels = testIds.map(allLabels)

    for ((net, users) <- normalizedData) {
      LOG.info("Evaluating " + net)
      val trainFeatures = makeFeaturesMatrix(users, trainIds)
      val testFeatures = makeFeaturesMatrix(users, testIds)

      val trainInstances = DataTransformer.constructInstances(trainFeatures, GENDER_VALUES, trainLabels)
      val testInstances = DataTransformer.constructInstances(testFeatures, GENDER_VALUES, testLabels)

      val relation = getRelation(trainInstances, testInstances)
      if (relation.nonEmpty) {
        val (k, fMeasure) = relation.maxBy(_._2)
        LOG.info(s"Best solution for $net: F-measure=$fMeasure (k=$k)")

        val relationFile = new File(singleLayerPCARelationsDir, net + ".csv")
        IO.writeRelation(Seq("k", "F-measure"), relation, relationFile)
      } else {
        LOG.warn("Relation is empty")
      }
    }
  }
}
