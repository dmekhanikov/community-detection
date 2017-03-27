package megabyte.communities.experiments.transformer

import java.io.File

import com.typesafe.scalalogging.Logger
import megabyte.communities.experiments.clustering.MultilayerSimilarityGraphClustering
import megabyte.communities.experiments.config.ExperimentConfig.config._
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

  private val graphFile = new File(baseDir, s"$city/graphs/similarity/twitter.csv")
  private val trainFile = new File(labelsDir, "train.arff")
  private val testFile = new File(labelsDir, "test.arff")

  private val GENDER_VALUES = Seq("male", "female")

  def main(args: Array[String]): Unit = {
    val (trainInstances, testInstances) = dataset()
    LOG.info(s"Writing arff file with train data to $trainFile")
    IO.writeInstances(trainInstances, trainFile)
    LOG.info(s"Writing arff file with test data to $testFile")
    IO.writeInstances(testInstances, testFile)
  }

  private def dataset(): (Instances, Instances) = {
    val allLabels = IO.readCSV(labelsFile)
      .map(m => m(ID_COL) -> m(GENDER_COL))
      .filter { case (_, v) => v.nonEmpty }
      .toMap

    val numeration = readNumeration(graphFile).filter(allLabels.contains)
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
