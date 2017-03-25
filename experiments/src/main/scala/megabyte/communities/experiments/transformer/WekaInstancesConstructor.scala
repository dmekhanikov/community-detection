package megabyte.communities.experiments.transformer

import java.io.File

import com.typesafe.scalalogging.Logger
import megabyte.communities.experiments.clustering.MultilayerSimilarityGraphClustering
import megabyte.communities.experiments.config.ExperimentConfig
import megabyte.communities.util.{DataTransformer, IO}
import org.jblas.DoubleMatrix
import weka.core.Instances

import scala.collection.GenSeq
import scala.util.Random

private class WekaInstancesConstructor

object WekaInstancesConstructor {

  private val LOG = Logger[WekaInstancesConstructor]

  private val ID_COL = "row ID"
  private val GENDER_COL = "gender"
  private val testFraction = 0.1

  private val BASE_DIR = ExperimentConfig.config.baseDir
  private val CITY = ExperimentConfig.config.city
  private val GRAPH_FILE = new File(BASE_DIR, s"$CITY/graphs/similarity/twitter.csv")
  private val LABELS_DIR = new File(BASE_DIR, s"$CITY/labels")
  private val LABELS_FILE = new File(LABELS_DIR, s"${CITY}GroundTruth.csv")
  private val TRAIN_FILE = new File(LABELS_DIR, "train.arff")
  private val TEST_FILE = new File(LABELS_DIR, "test.arff")

  private val GENDER_VALUES = Seq("male", "female")

  def main(args: Array[String]): Unit = {
    val (trainInstances, testInstances) = dataset()
    LOG.info(s"Writing arff file with train data to $TRAIN_FILE")
    IO.writeInstances(trainInstances, TRAIN_FILE)
    LOG.info(s"Writing arff file with test data to $TEST_FILE")
    IO.writeInstances(testInstances, TEST_FILE)
  }

  private def dataset(): (Instances, Instances) = {
    val allLabels = IO.readCSV(LABELS_FILE)
      .map(m => m(ID_COL) -> m(GENDER_COL))
      .filter { case (_, v) => v.nonEmpty }
      .toMap

    val numeration = readNumeration(GRAPH_FILE).filter(allLabels.contains)
    val permutation = Random.shuffle[Int, IndexedSeq](numeration.indices)
    val testSize = (numeration.size * testFraction).toInt
    val testIndices = permutation.take(testSize)
    val trainIndices = permutation.drop(testSize)

    val getLabels = { indices: GenSeq[Int] =>
      indices
        .map(i => numeration(i))
        .map(id => allLabels(id))
    }
    val trainLabels = getLabels(trainIndices)
    val testLabels = getLabels(testIndices)

    val allFeatures = featuresMatrix()
    val getFeatures = { indices: GenSeq[Int] => allFeatures.getRows(indices.toArray) }
    val trainFeatures = getFeatures(trainIndices)
    val testFeatures = getFeatures(testIndices)

    val trainInstances = DataTransformer.constructInstances(trainFeatures, GENDER_VALUES, trainLabels)
    val testInstances = DataTransformer.constructInstances(testFeatures, GENDER_VALUES, testLabels)
    (trainInstances, testInstances)
  }

  private def featuresMatrix(): DoubleMatrix =
    MultilayerSimilarityGraphClustering.subspace()

  private def readNumeration(graphFile: File): Seq[String] = {
    val source = io.Source.fromFile(graphFile)
    try {
      source.getLines().next().split(',')
    } finally {
      source.close()
    }
  }
}
