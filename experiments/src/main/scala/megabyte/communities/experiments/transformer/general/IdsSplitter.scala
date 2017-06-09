package megabyte.communities.experiments.transformer.general

import megabyte.communities.experiments.config.ExperimentConfig.config._
import megabyte.communities.util.IO

import scala.util.Random

object IdsSplitter {

  private val trainRatio = 0.8

  def main(args: Array[String]): Unit = {
    val instances = IO.readInstances(dataFile)
    val n = instances.size
    val trainN = (n * trainRatio).toInt
    val allIndices = Random.shuffle[Int, IndexedSeq](0 until n)
      .map(_.toString)
    val (trainIds, testIds) = allIndices.splitAt(trainN)
    idsDir.mkdirs()
    IO.writeLines(trainIds, trainIdsFile)
    IO.writeLines(testIds, testIdsFile)
  }
}
