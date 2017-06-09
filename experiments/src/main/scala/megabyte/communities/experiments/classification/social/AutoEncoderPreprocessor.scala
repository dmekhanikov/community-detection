package megabyte.communities.experiments.classification.social

import com.typesafe.scalalogging.Logger
import megabyte.communities.experiments.util.DataUtil.Users
import org.deeplearning4j.nn.api.OptimizationAlgorithm
import org.deeplearning4j.nn.conf.NeuralNetConfiguration
import org.deeplearning4j.nn.conf.layers.{OutputLayer, RBM}
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork
import org.deeplearning4j.optimize.listeners.ScoreIterationListener
import org.nd4j.linalg.activations.Activation
import org.nd4j.linalg.dataset.DataSet
import org.nd4j.linalg.factory.Nd4j
import org.nd4j.linalg.lossfunctions.LossFunctions

trait AutoEncoderPreprocessor {

  private val LOG = Logger[AutoEncoderPreprocessor]

  private val seed = 123
  private val iterations = 1
  private val listenerFreq = iterations / 5

  def transform(users: Users): Users = {
    LOG.info("Building model")
    val featuresNum = users.head._2.size
    val conf = new NeuralNetConfiguration.Builder()
      .seed(seed)
      .iterations(iterations)
      .optimizationAlgo(OptimizationAlgorithm.LINE_GRADIENT_DESCENT)
      .list
      .layer(0, new RBM.Builder().nIn(featuresNum).nOut(150).lossFunction(LossFunctions.LossFunction.KL_DIVERGENCE).build)
      .layer(1, new RBM.Builder().nIn(150).nOut(100).lossFunction(LossFunctions.LossFunction.KL_DIVERGENCE).build)
      .layer(2, new RBM.Builder().nIn(100).nOut(50).lossFunction(LossFunctions.LossFunction.KL_DIVERGENCE).build) //encoding stops
      .layer(3, new RBM.Builder().nIn(50).nOut(100).lossFunction(LossFunctions.LossFunction.KL_DIVERGENCE).build) //decoding starts
      .layer(4, new RBM.Builder().nIn(100).nOut(150).lossFunction(LossFunctions.LossFunction.KL_DIVERGENCE).build)
      .layer(5, new OutputLayer.Builder(LossFunctions.LossFunction.MSE).activation(Activation.SIGMOID).nIn(150).nOut(featuresNum).build)
      .pretrain(true).backprop(true)
      .build()

    val model = new MultiLayerNetwork(conf)
    model.init()

    model.setListeners(new ScoreIterationListener(listenerFreq))

    LOG.info("Training model")

    for ((features, i) <- users.values.zipWithIndex) {
      LOG.info(s"Processing user #$i/${users.size}")
      val featuresArray = Nd4j.create(features.toArray)
      val dataSet = new DataSet(featuresArray, featuresArray)
      dataSet.normalize()
      model.fit(dataSet)
    }

    LOG.info("Transforming data")
    users.map { case (id, features) =>
      val featuresArray = Nd4j.create(features.toArray, 'c')
      val transformedFeaturesArray = model.activateSelectedLayers(0, 4, featuresArray)
      val transformedFeatures = transformedFeaturesArray.data().array().asInstanceOf[Array[Double]].toSeq
      (id, transformedFeatures)
    }
  }

  private def toDataSet(users: Users): DataSet = {
    val usersNum = users.size
    val featuresNum = users.head._2.size
    val featuresArray = Nd4j.create(users.values.reduce { (a, b) => a ++ b }.toArray, Array(usersNum, featuresNum), 'c')
    new DataSet(featuresArray, null)
  }
}
