package megabyte.communities.experiments.classification.social

import java.io.File

import com.typesafe.scalalogging.Logger
import megabyte.communities.algo.constraints.CustomConstraintsApplier
import megabyte.communities.algo.graph.ConstrainedSpectralClustering
import megabyte.communities.experiments.classification.Evaluator
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

    val allLabels: Map[String, String] = readLabels(groundFile, ID_COL, GENDER_COL)
    val trainLabels = trainIds.map(allLabels)
    val testLabels = testIds.map(allLabels)

    val trainIndices = trainIds.map(allIds.indexOf(_))
    val testIndices = testIds.map(allIds.indexOf(_))

    CustomConstraintsApplier.constructClosure(q)
    for ((net, adj) <- networks.zip(adjs)) {
      LOG.info("Processing " + net)
      val (knn, k, fMeasure) = tuneFeaturesNum(trainIndices, trainLabels, testIndices, testLabels, adj, q)
      LOG.info(s"Best solution for $net:")
      logResult(knn, k, fMeasure)
    }
  }

  private def tuneFeaturesNum(trainIndices: Seq[Int], trainLabels: Seq[String],
                              testIndices: Seq[Int], testLabels: Seq[String],
                              adj: DoubleMatrix, q: DoubleMatrix): (Int, Int, Double) = {
    val randomForest = new RandomForest
    val (bestKnn, bestK, _) =
      (for (knn <- 2 to 10) yield {
        LOG.info(s"evaluating knn=$knn")
        val u = calcAllFeatures(adj, q, knn)
        for (k <- 2 to math.min(100, u.columns)) yield {
          LOG.info(s"evaluating knn=$knn; k=$k")
          val allFeatures = u.prefixColumns(k)
          val trainInstances = DataTransformer.constructInstances(allFeatures, k, trainIndices, GENDER_VALUES, trainLabels)
          val fMeasure = Evaluator.crossValidate(randomForest, trainInstances)
          logResult(knn, k, fMeasure)
          (knn, k, fMeasure)
        }
      })
        .flatten
        .maxBy(_._3)

    val allFeatures = calcAllFeatures(adj, q, bestKnn)
    val trainInstances = DataTransformer.constructInstances(allFeatures, bestK, trainIndices, GENDER_VALUES, trainLabels)
    val testInstances = DataTransformer.constructInstances(allFeatures, bestK, testIndices, GENDER_VALUES, testLabels)
    val fMeasure = Evaluator.evaluate(randomForest, trainInstances, testInstances)
    (bestKnn, bestK, fMeasure)
  }

  private def calcAllFeatures(w: DoubleMatrix, q: DoubleMatrix, knn: Int): DoubleMatrix = {
    val constraintsApplier = new CustomConstraintsApplier(knn)
    val clustering = new ConstrainedSpectralClustering(constraintsApplier)
    clustering.toEigenspace(w, q)
  }

  private def logResult(knn: Double, k: Int, fMeasure: Double): Unit = {
    LOG.info(s"knn: $knn; k: $k; F-measure: $fMeasure")
  }
}
