package megabyte.communities.experiments.classification

import java.io.File
import java.util.Random

import com.typesafe.scalalogging.Logger
import megabyte.communities.experiments.config.ExperimentConfig.config._
import megabyte.communities.util.IO
import weka.classifiers.Evaluation
import weka.classifiers.trees.RandomForest

private class RandomForestClassification

object RandomForestClassification {

  private val LOG = Logger[RandomForestClassification]

  private val FOLDS = 10

  private val trainFile = new File(labelsDir, "train.arff")
  private val testFile = new File(labelsDir, "test.arff")

  def main(args: Array[String]): Unit = {
    val trainData = IO.readInstances(trainFile)
    val testData = IO.readInstances(testFile)
    trainData.setClassIndex(trainData.numAttributes() - 1)

    val randomForest = new RandomForest()
    val evaluation = new Evaluation(trainData)
    evaluation.crossValidateModel(randomForest, trainData, FOLDS, new Random())
    printStats(evaluation)
  }

  private def printStats(evaluation: Evaluation): Unit = {
    System.err.println(evaluation.toSummaryString("\nResults\n======\n", true))
    System.err.println(evaluation.toClassDetailsString)
    System.err.println("Results For Class -1- ")
    System.err.println("Precision=  " + evaluation.precision(0))
    System.err.println("Recall=  " + evaluation.recall(0))
    System.err.println("F-measure=  " + evaluation.fMeasure(0))
    System.err.println("Results For Class -2- ")
    System.err.println("Precision=  " + evaluation.precision(1))
    System.err.println("Recall=  " + evaluation.recall(1))
    System.err.println("F-measure=  " + evaluation.fMeasure(1))
  }
}
