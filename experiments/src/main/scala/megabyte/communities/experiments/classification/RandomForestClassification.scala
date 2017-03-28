package megabyte.communities.experiments.classification

import java.io.File

import com.typesafe.scalalogging.Logger
import megabyte.communities.experiments.config.ExperimentConfig.config._
import megabyte.communities.util.IO
import weka.classifiers.Evaluation
import weka.classifiers.trees.RandomForest
import weka.core.Instances

private class RandomForestClassification

object RandomForestClassification {

  private val LOG = Logger[RandomForestClassification]

  private val trainFile = new File(labelsDir, "train.arff")
  private val testFile = new File(labelsDir, "test.arff")

  def main(args: Array[String]): Unit = {
    val trainData = IO.readInstances(trainFile)
    val testData = IO.readInstances(testFile)
    trainData.setClassIndex(trainData.numAttributes() - 1)
    evaluate(trainData, testData)
  }

  def evaluate(trainData: Instances, testData: Instances): Double = {
    val randomForest = new RandomForest()
    randomForest.buildClassifier(trainData)
    val evaluation = new Evaluation(trainData)
    evaluation.evaluateModel(randomForest, testData)
    evaluation.unweightedMacroFmeasure()
  }
}
