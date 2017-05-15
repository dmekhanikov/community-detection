package megabyte.communities.experiments.classification

import java.io.File

import com.typesafe.scalalogging.Logger
import megabyte.communities.algo.constraints.CustomConstraintsApplier
import megabyte.communities.algo.graph.ConstrainedSpectralClustering
import megabyte.communities.experiments.config.ExperimentConfig.config._
import megabyte.communities.experiments.util.DataUtil._
import megabyte.communities.util.DoubleMatrixOps._
import megabyte.communities.util.IO.readMatrixWithHeader
import megabyte.communities.util.{DataTransformer, IO}
import org.jblas.DoubleMatrix
import weka.classifiers.trees.RandomForest

object SingleLayerConstrained {

  private val LOG = Logger[SingleLayerConstrained.type]

  private val adjs = networks.par.map(net =>
    readMatrixWithHeader(new File(constrainedGraphsDir, net + ".csv"))
  ).seq.unzip._2

  private val q = readMatrixWithHeader(new File(constrainedGraphsDir, "q.csv"))._2

  def main(args: Array[String]): Unit = {
    val allIds = readIds(networks.head)
    val trainIds = IO.readLines(trainIdsFile)
    val testIds = IO.readLines(testIdsFile)

    val allLabels: Map[String, String] = readLabels(labelsFile, ID_COL, GENDER_COL)
    val trainLabels = trainIds.map(allLabels)
    val testLabels = testIds.map(allLabels)

    val trainIndices = trainIds.map(allIds.indexOf(_))
    val testIndices = testIds.map(allIds.indexOf(_))

    for ((net, adj) <- networks.zip(adjs)) {
      LOG.info("Processing " + net)
      val relation = getRelation(trainIndices, trainLabels, testIndices, testLabels, adj, q)
      val (knn, k, fMeasure) = relation.maxBy(_._2)
      LOG.info(s"Best solution for $net:")
      logResult(knn, k, fMeasure)
    }
  }

  private def getRelation(trainIndices: Seq[Int], trainLabels: Seq[String],
                          testIndices: Seq[Int], testLabels: Seq[String],
                          adj: DoubleMatrix, q: DoubleMatrix): Seq[(Int, Int, Double)] = {
    val randomForest = new RandomForest
    (for (knn <- 3 to 30) yield {
      val constraintsApplier = new CustomConstraintsApplier(knn)
      val constrainedClustering = new ConstrainedSpectralClustering(constraintsApplier)

      LOG.info(s"evaluating knn=$knn")
      val u = constrainedClustering.toEigenspace(adj, q)
      for (k <- 2 to math.min(100, u.columns)) yield {
        LOG.info(s"evaluating knn=$knn; k=$k")
        val allFeatures = u.prefixColumns(k)
        val trainFeatures = allFeatures.getRows(trainIndices.toArray)
        val testFeatures = allFeatures.getRows(testIndices.toArray)

        val trainInstances = DataTransformer.constructInstances(trainFeatures, GENDER_VALUES, trainLabels)
        val testInstances = DataTransformer.constructInstances(testFeatures, GENDER_VALUES, testLabels)
        val fMeasure = Evaluator.evaluate(randomForest, trainInstances, testInstances)
        logResult(knn, k, fMeasure)
        (knn, k, fMeasure)
      }
    }).flatten
  }

  private def logResult(knn: Double, k: Int, fMeasure: Double): Unit = {
    LOG.info(s"knn: $knn; k: $k; F-measure: $fMeasure")
  }
}
