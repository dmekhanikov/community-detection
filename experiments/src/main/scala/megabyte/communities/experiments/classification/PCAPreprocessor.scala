package megabyte.communities.experiments.classification

import com.typesafe.scalalogging.Logger
import megabyte.communities.experiments.util.DataUtil.attributesPrefixFilter
import weka.attributeSelection.{AttributeSelection, PrincipalComponents, Ranker}
import weka.classifiers.meta.FilteredClassifier
import weka.classifiers.trees.RandomForest
import weka.core.Instances

trait PCAPreprocessor {

  private val LOG = Logger[PCAPreprocessor]

  protected def getRelation(trainInstances: Instances, testInstances: Instances): Seq[(Int, Double)] = {
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
