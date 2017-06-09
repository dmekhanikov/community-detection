package megabyte.communities.experiments.classification.general

import megabyte.communities.experiments.classification.Evaluator
import megabyte.communities.experiments.config.ExperimentConfig.config._
import megabyte.communities.experiments.util.DataUtil
import weka.attributeSelection.{AttributeSelection, PrincipalComponents, Ranker}
import weka.classifiers.trees.RandomForest
import weka.core.Instances

object PrincipalComponents {

  def main(args: Array[String]): Unit = {
    val (trainInstances, testInstances) = DataUtil.readAndSplit(dataFile, trainIdsFile, testIdsFile)

    val selector = trainPCA(trainInstances)
    val pcaTrainInstances = selector.reduceDimensionality(trainInstances)
    val pcaTestInstances = selector.reduceDimensionality(testInstances)
    val classifier = new RandomForest
    val evaluation = Evaluator.getEvaluation(classifier, pcaTrainInstances, pcaTestInstances)
    Evaluator.printDetailedStats(evaluation)
  }

  private def trainPCA(trainInstances: Instances): AttributeSelection = {
    val pca = new PrincipalComponents()
    val ranker = new Ranker
    val selector = new AttributeSelection
    selector.setSearch(ranker)
    selector.setEvaluator(pca)
    selector.SelectAttributes(trainInstances)
    selector
  }
}
