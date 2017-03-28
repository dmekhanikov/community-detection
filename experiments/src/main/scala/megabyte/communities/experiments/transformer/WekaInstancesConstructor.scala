package megabyte.communities.experiments.transformer

import java.io.File

import com.typesafe.scalalogging.Logger
import megabyte.communities.experiments.Util._
import megabyte.communities.experiments.clustering.MultilayerSimilarityGraphClustering
import megabyte.communities.experiments.config.ExperimentConfig.config._
import megabyte.communities.util.{DataTransformer, IO}
import org.jblas.DoubleMatrix
import weka.core.Instances

import scala.util.Random

private class WekaInstancesConstructor

object WekaInstancesConstructor {

  private val LOG = Logger[WekaInstancesConstructor]

  private val TEST_FRACTION = 0.1

  private val graphFile = new File(baseDir, s"$city/graphs/similarity/twitter.csv")
  private val trainFile = new File(labelsDir, "train.arff")
  private val testFile = new File(labelsDir, "test.arff")

  def main(args: Array[String]): Unit = {
    val (trainInstances, testInstances) = dataset()
    LOG.info(s"Writing arff file with train data to $trainFile")
    IO.writeInstances(trainInstances, trainFile)
    LOG.info(s"Writing arff file with test data to $testFile")
    IO.writeInstances(testInstances, testFile)
  }

  private def dataset(): (Instances, Instances) = {
    val allLabels = readLabels(labelsFile, ID_COL, GENDER_COL)

    val numeration = IO.readHeader(graphFile).filter(allLabels.contains)
    val permutation = Random.shuffle[Int, Seq](numeration.indices)
    val (testIndices, trainIndices) = split(permutation, TEST_FRACTION)

    val trainLabels = getLabels(trainIndices, numeration, allLabels)
    val testLabels = getLabels(testIndices, numeration, allLabels)

    val allFeatures = featuresMatrix()
    val trainFeatures = allFeatures.getRows(trainIndices.toArray)
    val testFeatures = allFeatures.getRows(testIndices.toArray)
    val trainInstances = DataTransformer.constructInstances(trainFeatures, GENDER_VALUES, trainLabels)
    val testInstances = DataTransformer.constructInstances(testFeatures, GENDER_VALUES, testLabels)

    (trainInstances, testInstances)
  }

  private def featuresMatrix(): DoubleMatrix =
    MultilayerSimilarityGraphClustering.subspace()
}
