package megabyte.communities.experiments.classification

import java.io.File

import com.typesafe.scalalogging.Logger
import megabyte.communities.algo.graph.MultilayerSpectralClustering
import megabyte.communities.experiments.Util._
import megabyte.communities.experiments.config.ExperimentConfig.config._
import megabyte.communities.util.IO.{readMatrixWithHeader, readOrCalcMatrix}
import megabyte.communities.util.{DataTransformer, Graphs, IO}
import org.jblas.DoubleMatrix

import scala.util.Random

class MultilayerTuning

object MultilayerTuning {

  private val LOG = Logger[MultilayerTuning]

  private val TEST_FRACTION = 0.1
  private val graphFile = new File(similarityGraphsDir, "twitter.csv")
  private val relationFile = new File(relationsDir, "multilayer.csv")
  private val NETWORKS = Seq(
    "foursquare",
    "twitter",
    "instagram")

  private val adjs = NETWORKS.par.map { fileName =>
    readMatrixWithHeader(new File(similarityGraphsDir, fileName + ".csv"))._2
  }.seq
  private val lSyms = adjs.map(Graphs.symLaplacian)
  private val us = NETWORKS.zip(lSyms)
    .map { case (net, l) =>
      val file = new File(subspaceDir, net + ".csv")
      readOrCalcMatrix(file) {
        MultilayerSpectralClustering.toEigenspace(l)
      }
    }

  def main(args: Array[String]): Unit = {
    val allLabels = readLabels(labelsFile, ID_COL, GENDER_COL)

    val numeration = IO.readHeader(graphFile).filter(allLabels.contains)
    val permutation = Random.shuffle[Int, Seq](numeration.indices)
    val (testIndices, trainIndices) = split(permutation, TEST_FRACTION)

    val testLabels = getLabels(testIndices, numeration, allLabels)
    val trainLabels = getLabels(trainIndices, numeration, allLabels)

    val relation = getRelation(trainIndices, trainLabels, testIndices, testLabels)
    IO.writeRelation(Seq("k", "alpha", "F-measure"), relation, relationFile)
    val (k, alpha, fMeasure) = relation.maxBy(_._3)
    LOG.info("Best solution:")
    logResult(k, alpha, fMeasure)
  }

  private def getRelation(trainIndices: Seq[Int], trainLabels: Seq[String],
                                    testIndices: Seq[Int], testLabels: Seq[String]): Seq[(Int, Double, Double)] = {
    for (k <- 2 to 100; alpha <- 0.1 to 1 by 0.1) yield {
      LOG.info(s"evaluating k=$k; alpha=$alpha")
      val allFeatures = featuresMatrix(k, alpha)
      val trainFeatures = allFeatures.getRows(trainIndices.toArray)
      val testFeatures = allFeatures.getRows(testIndices.toArray)

      val trainInstances = DataTransformer.constructInstances(trainFeatures, GENDER_VALUES, trainLabels)
      val testInstances = DataTransformer.constructInstances(testFeatures, GENDER_VALUES, testLabels)
      val fMeasure = RandomForestClassification.evaluate(trainInstances, testInstances)
      logResult(k, alpha, fMeasure)
      (k, alpha, fMeasure)
    }
  }

  private def logResult(k: Int, alpha: Double, fMeasure: Double): Unit = {
    LOG.info(s"k: $k; alpha: $alpha; F-measure: $fMeasure")
  }

  private def featuresMatrix(k: Int, alpha: Double): DoubleMatrix = {
    MultilayerSpectralClustering.toCommonEigenspace(us, lSyms, k, alpha)
  }
}
