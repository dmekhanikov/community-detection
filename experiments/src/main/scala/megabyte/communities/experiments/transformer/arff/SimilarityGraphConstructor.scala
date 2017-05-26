package megabyte.communities.experiments.transformer.arff

import java.io.File

import com.typesafe.scalalogging.Logger
import megabyte.communities.experiments.config.ExperimentConfig.config._
import megabyte.communities.util.IO
import megabyte.communities.util.DataTransformer._
import megabyte.communities.util.DoubleMatrixOps._

object SimilarityGraphConstructor {

  private val LOG = Logger[SimilarityGraphConstructor.type]

  private val SIGMA_FACTOR = 1.5
  private val arffFile = new File(datasetDir, "data.arff")
  private val graphFile = new File(datasetDir, "graph.csv")

  def main(args: Array[String]): Unit = {
    val instances = IO.readInstances(arffFile)
    val n = instances.size
    val f = instances.numAttributes - 1
    val objects: Seq[Seq[Double]] =
      for (i <- 0 until n) yield {
        for (j <- 0 until f) yield {
          instances.get(i).value(j)
        }
      }
    val normalizedObjects = normalizeFeatures(objects).toSeq
    LOG.info(s"Calculating weight matrix")
    val w = heatWeightMatrix(normalizedObjects, SIGMA_FACTOR)
    LOG.info(s"Writing result to file")
    w.write(graphFile)
  }
}
