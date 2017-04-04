package megabyte.communities.experiments.classification

import java.io.File

import com.typesafe.scalalogging.Logger
import megabyte.communities.experiments.config.ExperimentConfig.config._
import megabyte.communities.experiments.util.DataUtil._
import megabyte.communities.util.DoubleMatrixOps._
import megabyte.communities.util.{DataTransformer, Graphs, IO}
import org.jblas.DoubleMatrix
import weka.classifiers.trees.RandomForest

object SingleLayerSpectral {

  private val LOG = Logger[SingleLayerSpectral.type]

  private val singleLayerRelationsDir = new File(relationsDir, "single_layer_spectral")

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

    val allLabels: Map[String, String] = readLabels(labelsFile, ID_COL, GENDER_COL)
    val trainIndices = trainIds.map(id => allIds.indexOf(id))
    val testIndices = testIds.map(id => allIds.indexOf(id))

    val trainLabels = trainIds.map(allLabels)
    val testLabels = testIds.map(allLabels)

    for ((net, u) <- networks.zip(us)) {
      LOG.info("Evaluating " + net)
      val relation = getRelation(u, trainIndices, trainLabels, testIndices, testLabels)
      IO.writeRelation(Seq("k", "F-measure"), relation,
        new File(singleLayerRelationsDir, net + ".csv"))
      val (k, fMeasure) = relation.maxBy(_._2)
      LOG.info(s"Best solution for $net: F-measure=$fMeasure (k=$k)")
    }
  }

  private def getRelation(u: DoubleMatrix,
                          trainIndices: Seq[Int], trainLabels: Seq[String],
                          testIndices: Seq[Int], testLabels: Seq[String]): Seq[(Int, Double)] = {
    val randomForest = new RandomForest
    for (k <- 2 to 100) yield {
      LOG.info(s"evaluating k=$k")
      val allFeatures = u.prefixColumns(k)
      val trainFeatures = allFeatures.getRows(trainIndices.toArray)
      val testFeatures = allFeatures.getRows(testIndices.toArray)

      val trainInstances = DataTransformer.constructInstances(trainFeatures, GENDER_VALUES, trainLabels)
      val testInstances = DataTransformer.constructInstances(testFeatures, GENDER_VALUES, testLabels)
      val fMeasure = Evaluator.evaluate(randomForest, trainInstances, testInstances)
      LOG.info(s"F-measure=$fMeasure (k=$k)")
      (k, fMeasure)
    }
  }
}
