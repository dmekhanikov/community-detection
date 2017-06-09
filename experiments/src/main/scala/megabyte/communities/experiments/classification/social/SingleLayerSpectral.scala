package megabyte.communities.experiments.classification.social

import com.typesafe.scalalogging.Logger
import megabyte.communities.experiments.classification.Evaluator
import megabyte.communities.experiments.config.ExperimentConfig.config._
import megabyte.communities.experiments.util.DataUtil._
import megabyte.communities.util.DataTransformer.constructInstances
import megabyte.communities.util.{Graphs, IO}
import org.jblas.DoubleMatrix
import weka.classifiers.trees.RandomForest

object SingleLayerSpectral {

  private val LOG = Logger[SingleLayerSpectral.type]

  private val lSyms =
    networks.par.map(net => readAdj(net)._2).seq
      .map(Graphs.symLaplacian)
  private val us = networks.zip(lSyms).map { case (net, l) =>
    readOrCalcSymSubspace(net, l)
  }

  def main(args: Array[String]): Unit = {
    val allIds = readIds(networks.head)
    val trainIds = IO.readLines(trainIdsFile)
    val testIds = IO.readLines(testIdsFile)

    val allLabels: Map[String, String] = readLabels(groundFile, ID_COL, GENDER_COL)
    val trainIndices = trainIds.map(id => allIds.indexOf(id))
    val testIndices = testIds.map(id => allIds.indexOf(id))

    val trainLabels = trainIds.map(allLabels)
    val testLabels = testIds.map(allLabels)

    for ((net, u) <- networks.zip(us)) {
      LOG.info("Evaluating " + net)
      val (k, fMeasure) = tuneFeaturesNum(u, trainIndices, trainLabels, testIndices, testLabels)
      LOG.info(s"Best solution for $net: F-measure=$fMeasure (k=$k)")
    }
  }

  private def tuneFeaturesNum(u: DoubleMatrix,
                              trainIndices: Seq[Int], trainLabels: Seq[String],
                              testIndices: Seq[Int], testLabels: Seq[String]): (Int, Double) = {
    val randomForest = new RandomForest
    val bestK = (for (k <- 2 to math.min(100, u.columns)) yield {
      LOG.info(s"evaluating k=$k")
      val instances = constructInstances(u, k, trainIndices, GENDER_VALUES, trainLabels)
      val fMeasure = Evaluator.crossValidate(randomForest, instances)
      LOG.info(s"F-measure=$fMeasure (k=$k)")
      (k, fMeasure)
    })
      .maxBy(_._2)
      ._1
    val trainInstances = constructInstances(u, bestK, trainIndices, GENDER_VALUES, trainLabels)
    val testInstances = constructInstances(u, bestK, testIndices, GENDER_VALUES, testLabels)
    val fMeasure = Evaluator.evaluate(randomForest, trainInstances, testInstances)
    (bestK, fMeasure)
  }
}
