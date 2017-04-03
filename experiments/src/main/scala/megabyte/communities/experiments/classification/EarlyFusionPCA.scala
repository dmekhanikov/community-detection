package megabyte.communities.experiments.classification

import java.io.File

import com.typesafe.scalalogging.Logger
import megabyte.communities.experiments.config.ExperimentConfig.config._
import megabyte.communities.experiments.util.DataUtil._
import megabyte.communities.util.{DataTransformer, IO}
import weka.attributeSelection.{AttributeSelection, PrincipalComponents, Ranker}
import weka.classifiers.meta.FilteredClassifier
import weka.classifiers.trees.RandomForest
import weka.core.Instances

private class EarlyFusionPCA

object EarlyFusionPCA {

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

  private def getRelation(trainInstances: Instances, testInstances: Instances): Seq[(Int, Double)] = {
    val pca = new PrincipalComponents()
    val ranker = new Ranker
    val selector = new AttributeSelection
    selector.setSearch(ranker)
    selector.setEvaluator(pca)
    selector.SelectAttributes(trainInstances)
    val pcaTrainInstances = selector.reduceDimensionality(trainInstances)
    val pcaTestInstances = selector.reduceDimensionality(testInstances)

    val randomForest = new RandomForest
    val filteredClassifier = new FilteredClassifier
    (2 until pcaTrainInstances.numAttributes) map { k =>
      LOG.info(s"Evaluating k=$k")
      val filter = attributesPrefixFilter(k)
      filteredClassifier.setFilter(filter)
      filteredClassifier.setClassifier(randomForest)
      val fMeasure = Evaluator.evaluate(filteredClassifier, pcaTrainInstances, pcaTestInstances)
      LOG.info(s"F-measure for k=$k: $fMeasure")
      (k, fMeasure)
    }
  }
}
