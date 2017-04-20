package megabyte.communities.experiments.classification

import com.typesafe.scalalogging.Logger
import megabyte.communities.algo.graph.LuConstrainedSpectralClustering
import megabyte.communities.experiments.config.ExperimentConfig.config._
import megabyte.communities.experiments.util.DataUtil._
import megabyte.communities.util.{DataTransformer, IO}
import weka.classifiers.trees.RandomForest

object SingleLayerConstrained {

  private val LOG = Logger[SingleLayerConstrained.type]

  private val (networksHashes, adjs) = networks.par.map(readAdj).seq.unzip

  private val qs = networks.zip(networksHashes).par.map { case (net, hashes) =>
    val q = readConstraintsMatrix(s"$net.graphml", hashes)
    for (i <- 0 until q.rows; j <- 0 until q.columns) {
      q.put(i, j, q.get(i, j) / 2)
    }
    q
  }.seq

  def main(args: Array[String]): Unit = {
    val allIds = readIds(networks.head)
    val trainIds = IO.readLines(trainIdsFile)
    val testIds = IO.readLines(testIdsFile)

    val allLabels: Map[String, String] = readLabels(labelsFile, ID_COL, GENDER_COL)
    val trainLabels = trainIds.map(allLabels)
    val testLabels = testIds.map(allLabels)

    val trainIndices = trainIds.map(allIds.indexOf(_))
    val testIndices = testIds.map(allIds.indexOf(_))

    val randomForest = new RandomForest

    for ((net, (adj, q)) <- networks.zip(adjs.zip(qs))) {
      LOG.info("Processing " + net)
      for (knn <- 5 to 50 by 5; alpha <- 0.05 to 1 by 0.05) {
        val allFeatures = LuConstrainedSpectralClustering.toEigenspace(adj, q, knn, alpha)
        val trainFeatures = allFeatures.getRows(trainIndices.toArray)
        val testFeatures = allFeatures.getRows(testIndices.toArray)

        val trainInstances = DataTransformer.constructInstances(trainFeatures, GENDER_VALUES, trainLabels)
        val testInstances = DataTransformer.constructInstances(testFeatures, GENDER_VALUES, testLabels)

        val fMeasure = Evaluator.evaluate(randomForest, trainInstances, testInstances)
        LOG.info(s"Results for $net: F-measure=$fMeasure (knn=$knn; alpha=$alpha)")
      }
    }
  }
}
