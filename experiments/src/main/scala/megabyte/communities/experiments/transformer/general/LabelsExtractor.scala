package megabyte.communities.experiments.transformer.general

import megabyte.communities.util.IO

import megabyte.communities.experiments.config.ExperimentConfig.config._

import scala.collection.JavaConversions._

object LabelsExtractor {

  def main(args: Array[String]): Unit = {
    val instances = IO.readInstances(dataFile)
    instances.setClassIndex(instances.numAttributes - 1)
    val labels = instances.map(_.classValue().round.toString)
    IO.writeLines(labels, labelsFile)
  }
}
