package megabyte.communities.experiments.classification.social

import com.typesafe.scalalogging.Logger
import megabyte.communities.experiments.classification.general.Spectral
import megabyte.communities.experiments.config.ExperimentConfig.config._
import megabyte.communities.experiments.util.DataUtil._
import megabyte.communities.util.{Graphs, IO}
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

    val labels: Map[String, String] = readLabels(groundFile, ID_COL, GENDER_COL)
    val trainIndices = trainIds.map(id => allIds.indexOf(id))
    val testIndices = testIds.map(id => allIds.indexOf(id))
    val labelsSeq = allIds.map(labels)

    for ((net, u) <- networks.zip(us)) {
      LOG.info("Evaluating " + net)
      val (k, fMeasure) = Spectral.tuneFeaturesNum(u, trainIndices, testIndices, labelsSeq, new RandomForest)
      LOG.info(s"Best solution for $net: F-measure=$fMeasure (k=$k)")
    }
  }
}
