package megabyte.communities.experiments.classification

import com.typesafe.scalalogging.Logger
import jsat.classifiers.{CategoricalData, ClassificationDataSet, DataPoint}
import jsat.datatransform.FastICA
import jsat.linear.DenseVector
import megabyte.communities.experiments.config.ExperimentConfig.config._
import megabyte.communities.experiments.util.DataUtil._
import megabyte.communities.util.{DataTransformer, IO}
import weka.classifiers.trees.RandomForest

import scala.collection.JavaConversions._

object SingleLayerICA {

  private val LOG = Logger[SingleLayerICA.type]

  def main(args: Array[String]): Unit = {
    val networkUsers: Map[String, Seq[Users]] =
      featureFiles.mapValues(_.map(readUsersData)).view.force
    val mergedUsers: Map[String, Users] = mergeUserData(networkUsers)
    val normalizedData: Map[String, Users] = mergedUsers.mapValues(normalizeUserFeatures)

    val trainIds = IO.readLines(trainIdsFile)
    val testIds = IO.readLines(testIdsFile)

    val labels: Map[String, String] = readLabels(labelsFile, ID_COL, GENDER_COL)

    for ((net, users) <- normalizedData) {
      LOG.info("Evaluating " + net)
      val (componentNum, _) = tuneComponentsNum(users, labels, trainIds)

      val trainSet = constructDataSet(users, labels, trainIds)
      val testSet = constructDataSet(users, labels, testIds)
      val fastICA = new FastICA(componentNum)
      fastICA.fit(trainSet)
      trainSet.applyTransform(fastICA, true)
      testSet.applyTransform(fastICA, true)

      val trainInstances = DataTransformer.constructInstances(trainSet)
      val testInstances = DataTransformer.constructInstances(testSet)
      val evaluation = Evaluator.getEvaluation(new RandomForest, trainInstances, testInstances)
      val fMeasure = evaluation.unweightedMacroFmeasure()
      LOG.info(s"Best solution for $net:")
      LOG.info(s"Components: $componentNum; F-Measure: $fMeasure")
    }
  }

  private def tuneComponentsNum(users: Users, labels: Map[String, String], ids: Seq[String]): (Int, Double) = {
    val randomForest = new RandomForest
    val relation = for (componentNum <- 2 to 100) yield {
      LOG.info(s"Trying $componentNum components")
      val dataSet = constructDataSet(users, labels, ids)

      val fastICA = new FastICA(componentNum)
      LOG.info("Training ICA")
      fastICA.fit(dataSet)
      LOG.info("Transforming data")
      dataSet.applyTransform(fastICA, true)

      val instances = DataTransformer.constructInstances(dataSet)
      val evaluation = Evaluator.getCVEvaluation(randomForest, instances)
      val fMeasure = evaluation.unweightedMacroFmeasure()
      LOG.info(s"Components: $componentNum; F-Measure: $fMeasure")
      (componentNum, fMeasure)
    }
    relation.maxBy(_._2)
  }

  private def constructDataSet(users: Users, labels: Map[String, String], ids: Seq[String]): ClassificationDataSet = {
    val genderCatData = new CategoricalData(GENDER_VALUES.size)
    val dataPoints = ids.map { id =>
        val features: Features = users(id)
        val label = labels(id)
        val featuresVec = new DenseVector(features.toArray)
        val genderValue = GENDER_VALUES.indexOf(label)
        new DataPoint(featuresVec, Array(genderValue), Array(genderCatData))
      }
    new ClassificationDataSet(dataPoints, 0)
  }
}
