package megabyte.communities.experiments.classification

import java.io.File

import com.typesafe.scalalogging.Logger
import megabyte.communities.experiments.config.ExperimentConfig.config._
import megabyte.communities.util.IO
import weka.classifiers.trees.RandomForest
import weka.classifiers.{Classifier, Evaluation}
import weka.core.Instances

object Evaluator {

  private val LOG = Logger[Evaluator.type]

  private val trainFile = new File(labelsDir, "train.arff")
  private val testFile = new File(labelsDir, "test.arff")

  def main(args: Array[String]): Unit = {
    val trainData = IO.readInstances(trainFile)
    val testData = IO.readInstances(testFile)
    trainData.setClassIndex(trainData.numAttributes() - 1)
    testData.setClassIndex(trainData.numAttributes() - 1)
    val randomForest = new RandomForest
    val evaluation = getEvaluation(randomForest, trainData, testData)
    printDetailedStats(evaluation)
  }

  def evaluate(classifier: Classifier, trainData: Instances, testData: Instances): Double = {
    val evaluation = getEvaluation(classifier, trainData, testData)
    evaluation.unweightedMacroFmeasure()
  }

  def getEvaluation(classifier: Classifier, trainData: Instances, testData: Instances): Evaluation = {
    classifier.buildClassifier(trainData)
    val evaluation = new Evaluation(trainData)
    evaluation.evaluateModel(classifier, testData)
    evaluation
  }

  def printDetailedStats(evaluation: Evaluation): Unit = {
    LOG.info(evaluation.toSummaryString("Results", true))
    LOG.info(evaluation.toClassDetailsString)
    LOG.info("Micro F-measure = " + evaluation.unweightedMicroFmeasure())
    LOG.info("Macro F-measure = " + evaluation.unweightedMacroFmeasure())
    LOG.info("Results For Class -1-")
    LOG.info("Precision = " + evaluation.precision(0))
    LOG.info("Recall = " + evaluation.recall(0))
    LOG.info("F-measure = " + evaluation.fMeasure(0))
    LOG.info("Results For Class -2-")
    LOG.info("Precision = " + evaluation.precision(1))
    LOG.info("Recall = " + evaluation.recall(1))
    LOG.info("F-measure = " + evaluation.fMeasure(1))
  }
}
