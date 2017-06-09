package megabyte.communities.experiments.transformer.social

import java.io.File

import com.typesafe.scalalogging.Logger
import megabyte.communities.experiments.config.ExperimentConfig.config._
import megabyte.communities.experiments.util.DataUtil._
import megabyte.communities.util.DataTransformer.heatWeightMatrix
import megabyte.communities.util.DoubleMatrixOps._
import megabyte.communities.util.IO
import org.jblas.DoubleMatrix

object ConstrainedSimilarityGraphConstructor {

  private val LOG = Logger[ConstrainedSimilarityGraphConstructor.type]

  def main(args: Array[String]): Unit = {
        val allLabels: Map[String, String] = readLabels(groundFile, ID_COL, GENDER_COL)
    val networksData: Map[String, Seq[Users]] =
      featureFiles.mapValues(_.map(readUsersData)).view.force
    val mergedData: Map[String, Users] = mergeUserData(networksData)
    val normalizedData: Map[String, Users] = mergedData.mapValues(normalizeUserFeatures)
    val numeration: Seq[String] = mergedData.values.flatMap(_.keys).toSet.toSeq
    val fullNumeration = numeration ++ Seq("male", "female")
    val trainIds = IO.readLines(trainIdsFile)

    constrainedGraphsDir.mkdirs()
    normalizedData.par.foreach { case (net, users) =>
      LOG.info(s"Calculating adjacency matrix for $net")
      val meanMale = calcLabelMean(users, allLabels, trainIds, "male")
      val meanFemale = calcLabelMean(users, allLabels, trainIds, "female")
      val meanMap = Map("male" -> meanMale, "female" -> meanFemale)
      val allUsers = users ++ meanMap
      val objects = fullNumeration.map(allUsers)
      val adj = heatWeightMatrix(objects, SimilarityGraphConstructor.SIGMA_FACTOR)
      val outFile = new File(constrainedGraphsDir, s"$net.csv")
      LOG.info(s"Writing result for $net to $outFile")
      adj.write(outFile, header = Some(fullNumeration))
    }

    val q = makeConstraintsMatrix(fullNumeration)
    q.write(new File(constrainedGraphsDir, "q.csv"), header = Some(fullNumeration))
    LOG.info("Finished, exiting")
  }

  private def makeConstraintsMatrix(numeration: Seq[String]): DoubleMatrix = {
    LOG.info("Constructing constraints matrix")
    val n = numeration.size
    val q = new DoubleMatrix(n, n)
    val maleSubscribers = IO.readLines(new File(portalsDir, "maleSubscribers.txt")).map(md5)
    val femaleSubscribers = IO.readLines(new File(portalsDir, "femaleSubscribers.txt")).map(md5)
    val maleId = n - 2
    val femaleId = n - 1
    connectSubscribers(q, numeration, maleSubscribers, maleId)
    connectSubscribers(q, numeration, femaleSubscribers, femaleId)
    q
  }

  private def connectSubscribers(m: DoubleMatrix, numeration: Seq[String], ids: Seq[String], portalPos: Int): Unit = {
    LOG.info(s"Connecting subscribers of ${numeration(portalPos)}")
    val invNumeration = numeration.zipWithIndex.map { case (id, i) =>
      id -> i
    }.toMap
    for (id <- ids) {
      val posOpt = invNumeration.get(id)
      posOpt match {
        case Some(pos) =>
          posOpt.foreach { pos =>
            m.put(pos, portalPos, 1)
            m.put(portalPos, pos, 1)
          }
        case None =>
      }
    }
  }

  private def calcLabelMean(users: Users, labels: Map[String, String], ids: Seq[String], target: String): Features = {
    val labelIds = ids.filter(id => labels.get(id).contains(target))
    val labelUsers = labelIds.map(id => users(id))
    val sumFeatures = labelUsers.reduce((a, b) =>
      a.zip(b).map { case (x, y) => x + y})
    sumFeatures.map(f => f / labelIds.size)
  }
}
