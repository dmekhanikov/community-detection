package megabyte.communities.experiments.classification.general

import com.typesafe.scalalogging.Logger
import megabyte.communities.algo.graph.SpectralClustering
import megabyte.communities.experiments.classification.Evaluator
import megabyte.communities.experiments.config.ExperimentConfig.config._
import megabyte.communities.util.DataTransformer.constructInstances
import megabyte.communities.util.{Graphs, IO}
import org.jblas.DoubleMatrix
import weka.classifiers.Classifier
import weka.classifiers.trees.RandomForest

object Spectral {

  private val LOG = Logger[Spectral.type]

  def main(args: Array[String]): Unit = {
    val adj = IO.readMatrix(graphFile)
    val lSym = Graphs.symLaplacian(adj)
    val u = SpectralClustering.toEigenspace(lSym)
    val trainIds = IO.readLines(trainIdsFile).map(_.toInt)
    val testIds = IO.readLines(testIdsFile).map(_.toInt)
    val labels = IO.readLines(labelsFile)
    val classifier = new RandomForest
    val (k, fMeasure) = tuneFeaturesNum(u, trainIds, testIds, labels, classifier)
    LOG.info("Best solution:")
    LOG.info(s"k=$k; F-measure=$fMeasure")
  }

  def tuneFeaturesNum(u: DoubleMatrix,
                      trainIndices: Seq[Int],
                      testIndices: Seq[Int],
                      labels: Seq[String],
                      classifier: Classifier): (Int, Double) = {
    val labelValues = labels.distinct
    val trainLabels = trainIndices.map(labels)
    val testLabels = testIndices.map(labels)
    val bestK = (for (k <- 2 to math.min(100, u.columns)) yield {
      LOG.info(s"evaluating k=$k")
      val instances = constructInstances(u, k, trainIndices, labelValues, trainLabels)
      val fMeasure = Evaluator.crossValidate(classifier, instances)
      LOG.info(s"F-measure=$fMeasure (k=$k)")
      (k, fMeasure)
    })
      .maxBy(_._2)
      ._1
    val trainInstances = constructInstances(u, bestK, trainIndices, labelValues, trainLabels)
    val testInstances = constructInstances(u, bestK, testIndices, labelValues, testLabels)
    val fMeasure = Evaluator.evaluate(classifier, trainInstances, testInstances)
    (bestK, fMeasure)
  }
}
