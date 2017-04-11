package megabyte.communities.experiments.classification

import java.io.File

import com.typesafe.scalalogging.Logger
import megabyte.communities.algo.graph.MultilayerSpectralClustering
import megabyte.communities.experiments.config.ExperimentConfig.config._
import megabyte.communities.experiments.util.DataUtil._
import megabyte.communities.util.{DataTransformer, Graphs, IO}
import org.jblas.DoubleMatrix
import weka.classifiers.trees.RandomForest

object MultilayerSpectral {

  private val LOG = Logger[MultilayerSpectral.type]

  def main(args: Array[String]): Unit = {
    val relationFile = new File(relationsDir, "multilayer.csv")
    val lSyms =
      networks.par.map(net => readAdj(net)._2).seq
        .map(Graphs.symLaplacian)
    val us = networks.zip(lSyms).map {
      case (net, l) => readOrCalcSymSubspace(net, l)
    }
    run(lSyms, us, relationFile)
  }

  def run(lSyms: Seq[DoubleMatrix],
          us: Seq[DoubleMatrix],
          relationFile: File): Unit = {
    val allIds = readIds(networks.head)
    val trainIds = IO.readLines(trainIdsFile)
    val testIds = IO.readLines(testIdsFile)

    val allLabels: Map[String, String] = readLabels(labelsFile, ID_COL, GENDER_COL)
    val trainLabels = trainIds.map(allLabels)
    val testLabels = testIds.map(allLabels)

    val trainIndices = trainIds.map(id => allIds.indexOf(id))
    val testIndices = testIds.map(id => allIds.indexOf(id))

    val relation = getRelation(trainIndices, trainLabels, testIndices, testLabels, lSyms, us)
    IO.writeRelation(Seq("k", "alpha", "F-measure"), relation, relationFile)
    val (k, alpha, fMeasure) = relation.maxBy(_._3)
    LOG.info("Best solution:")
    logResult(k, alpha, fMeasure)
  }

  private def getRelation(trainIndices: Seq[Int], trainLabels: Seq[String],
                          testIndices: Seq[Int], testLabels: Seq[String],
                          lSyms: Seq[DoubleMatrix], us: Seq[DoubleMatrix]): Seq[(Int, Double, Double)] = {
    val randomForest = new RandomForest
    for (k <- 2 to 100; alpha <- 0.1 to 1 by 0.1) yield {
      LOG.info(s"evaluating k=$k; alpha=$alpha")
      val allFeatures = MultilayerSpectralClustering.toCommonEigenspace(lSyms, us, k, alpha)
      val trainFeatures = allFeatures.getRows(trainIndices.toArray)
      val testFeatures = allFeatures.getRows(testIndices.toArray)

      val trainInstances = DataTransformer.constructInstances(trainFeatures, GENDER_VALUES, trainLabels)
      val testInstances = DataTransformer.constructInstances(testFeatures, GENDER_VALUES, testLabels)
      val fMeasure = Evaluator.evaluate(randomForest, trainInstances, testInstances)
      logResult(k, alpha, fMeasure)
      (k, alpha, fMeasure)
    }
  }

  private def logResult(k: Int, alpha: Double, fMeasure: Double): Unit = {
    LOG.info(s"k: $k; alpha: $alpha; F-measure: $fMeasure")
  }
}
