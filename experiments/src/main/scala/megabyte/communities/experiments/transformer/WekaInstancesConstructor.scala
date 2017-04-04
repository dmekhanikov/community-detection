package megabyte.communities.experiments.transformer

import java.io.File

import com.typesafe.scalalogging.Logger
import megabyte.communities.experiments.clustering.MultilayerSimilarityGraphClustering
import megabyte.communities.experiments.config.ExperimentConfig.config._
import megabyte.communities.experiments.util.DataUtil._
import megabyte.communities.util.{DataTransformer, IO}
import org.jblas.DoubleMatrix
import weka.core.Instances

private class WekaInstancesConstructor

object WekaInstancesConstructor {

  private val LOG = Logger[WekaInstancesConstructor]

  private val graphFile = new File(similarityGraphsDir, "twitter.csv")
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

    val trainIds = IO.readLines(trainIdsFile)
    val testIds = IO.readLines(testIdsFile)

    val allIds = IO.readHeader(graphFile)
    val trainIndices = trainIds.map(id => allIds.indexOf(id))
    val testIndices = testIds.map(id => allIds.indexOf(id))

    val trainLabels = trainIds.map(allLabels)
    val testLabels = testIds.map(allLabels)

    val allFeatures = featuresMatrix()
    val trainFeatures = allFeatures.getRows(trainIndices.toArray)
    val testFeatures = allFeatures.getRows(testIndices.toArray)
    val trainInstances = DataTransformer.constructInstances(trainFeatures, GENDER_VALUES, trainLabels)
    val testInstances = DataTransformer.constructInstances(testFeatures, GENDER_VALUES, testLabels)

    (trainInstances, testInstances)
  }

  private def featuresMatrix(): DoubleMatrix =
    MultilayerSimilarityGraphClustering.subspace(10, 0.2)
}
