package megabyte.communities.experiments.classification.general

import megabyte.communities.experiments.classification.Evaluator
import megabyte.communities.experiments.config.ExperimentConfig.config._
import megabyte.communities.experiments.util.DataUtil
import weka.classifiers.trees.RandomForest

object Simple {

  def main(args: Array[String]): Unit = {
    val (trainInstances, testInstances) = DataUtil.readAndSplit(dataFile, trainIdsFile, testIdsFile)
    val classifier = new RandomForest
    val evaluation = Evaluator.getEvaluation(classifier, trainInstances, testInstances)
    Evaluator.printDetailedStats(evaluation)
  }
}
