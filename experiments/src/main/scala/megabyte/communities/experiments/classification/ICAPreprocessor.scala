package megabyte.communities.experiments.classification

import com.typesafe.scalalogging.Logger
import jsat.classifiers.{CategoricalData, ClassificationDataSet, DataPoint}
import jsat.datatransform.FastICA
import jsat.linear.DenseVector
import megabyte.communities.experiments.classification.EarlyFusionICA.{constructDataSet, tuneComponentsNum}
import megabyte.communities.experiments.util.DataUtil.{Features, GENDER_VALUES, Users}
import megabyte.communities.util.DataTransformer
import weka.classifiers.trees.RandomForest

import scala.collection.JavaConversions._

trait ICAPreprocessor {

  private val LOG = Logger[ICAPreprocessor]

  def tuneComponentsNum(users: Users, labels: Map[String, String], ids: Seq[String]): (Int, Double) = {
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

  def constructDataSet(users: Users, labels: Map[String, String], ids: Seq[String]): ClassificationDataSet = {
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

  def evaluate(users: Users, labels: Map[String, String], trainIds: Seq[String], testIds: Seq[String], componentNum: Int): Double = {
    val trainSet = constructDataSet(users, labels, trainIds)
    val testSet = constructDataSet(users, labels, testIds)
    val fastICA = new FastICA(componentNum)
    fastICA.fit(trainSet)
    trainSet.applyTransform(fastICA, true)
    testSet.applyTransform(fastICA, true)

    val trainInstances = DataTransformer.constructInstances(trainSet)
    val testInstances = DataTransformer.constructInstances(testSet)
    val evaluation = Evaluator.getEvaluation(new RandomForest, trainInstances, testInstances)
    evaluation.unweightedMacroFmeasure()
  }
}
