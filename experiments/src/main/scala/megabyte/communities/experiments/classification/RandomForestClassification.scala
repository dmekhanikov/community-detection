package megabyte.communities.experiments.classification

import java.io.File
import java.util.Random

import com.typesafe.scalalogging.Logger
import megabyte.communities.experiments.config.ExperimentConfig
import megabyte.communities.util.IO
import weka.classifiers.Evaluation
import weka.classifiers.trees.RandomForest

private class RandomForestClassification

object RandomForestClassification {

  private val LOG = Logger[RandomForestClassification]

  private val FOLDS = 10

  private val BASE_DIR = ExperimentConfig.config.baseDir
  private val CITY = ExperimentConfig.config.city
  private val LABELS_DIR = new File(s"$BASE_DIR/$CITY/labels")
  private val TRAIN_FILE = new File(LABELS_DIR, "train.arff")
  private val TEST_FILE = new File(LABELS_DIR, "test.arff")

  def main(args: Array[String]): Unit = {
    val trainData = IO.readInstances(TRAIN_FILE)
    val testData = IO.readInstances(TEST_FILE)
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
