package megabyte.communities.experiments.classification.general

import com.typesafe.scalalogging.Logger
import megabyte.communities.experiments.classification.Evaluator
import megabyte.communities.experiments.config.ExperimentConfig.config._
import megabyte.communities.experiments.util.DataUtil
import megabyte.communities.experiments.util.DataUtil.attributesPrefixFilter
import weka.attributeSelection
import weka.attributeSelection.{AttributeSelection, Ranker}
import weka.classifiers.Classifier
import weka.classifiers.meta.FilteredClassifier
import weka.classifiers.trees.RandomForest
import weka.core.Instances

object PrincipalComponents {

  private val LOG = Logger[PrincipalComponents.type]

  def main(args: Array[String]): Unit = {
    val (trainInstances, testInstances) = DataUtil.readAndSplit(dataFile, trainIdsFile, testIdsFile)
    val classifier = new RandomForest
    val (k, fMeasure) = tuneFeaturesNum(trainInstances, testInstances, classifier)
    LOG.info("Best solution:")
    LOG.info(s"k=$k; F-measure=$fMeasure")
  }

  private def trainPCA(trainInstances: Instances): AttributeSelection = {
    val pca = new attributeSelection.PrincipalComponents()
    val ranker = new Ranker
    val selector = new AttributeSelection
    selector.setSearch(ranker)
    selector.setEvaluator(pca)
    selector.SelectAttributes(trainInstances)
    selector
  }

  def tuneFeaturesNum(trainInstances: Instances, testInstances: Instances, classifier: Classifier): (Int, Double) = {
    val selector = trainPCA(trainInstances)
    val pcaTrainInstances = selector.reduceDimensionality(trainInstances)
    val pcaTestInstances = selector.reduceDimensionality(testInstances)

    val filteredClassifier = new FilteredClassifier
    val relation = for (k <- 2 until math.min(100, pcaTrainInstances.numAttributes)) yield {
      LOG.info(s"Evaluating k=$k")
      val filter = attributesPrefixFilter(k)
      filteredClassifier.setFilter(filter)
      filteredClassifier.setClassifier(classifier)
      val fMeasure = Evaluator.crossValidate(filteredClassifier, pcaTrainInstances)
      LOG.info(s"F-measure for k=$k: $fMeasure")
      (k, fMeasure)
    }
    if (relation.isEmpty) {
      LOG.warn("Relation is empty")
      (0, Double.NaN)
    } else {
      val bestK = relation.maxBy(_._2)._1
      val filter = attributesPrefixFilter(bestK)
      filteredClassifier.setFilter(filter)
      filteredClassifier.setClassifier(classifier)
      val fMeasure = Evaluator.evaluate(filteredClassifier, pcaTrainInstances, pcaTestInstances)
      (bestK, fMeasure)
    }
  }
}
