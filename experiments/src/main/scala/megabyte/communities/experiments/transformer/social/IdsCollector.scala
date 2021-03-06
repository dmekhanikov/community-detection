package megabyte.communities.experiments.transformer.social

import java.io.File

import megabyte.communities.experiments.config.ExperimentConfig.config._
import megabyte.communities.util.IO

object IdsCollector {

  def main(args: Array[String]): Unit = {
    val idSets = for (dataset <- Seq("London", "NewYork", "Singapore")) yield {
      readIds(new File(baseDir, s"$dataset/ids/twitter.csv")).toSet
    }
    val numIds = idSets.reduce((a, b) => a.union(b)).toSeq
    IO.writeLines(numIds, allIdsFile)
  }

  private def readIds(file: File): Seq[String] = {
    IO.readCSVToSeq(file).map(_.head)
  }
}
