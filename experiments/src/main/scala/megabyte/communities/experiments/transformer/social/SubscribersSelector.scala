package megabyte.communities.experiments.transformer.social

import java.io.File

import megabyte.communities.experiments.config.ExperimentConfig.config._
import megabyte.communities.experiments.dao.MongoDAO
import megabyte.communities.experiments.util.DataUtil.{GENDER_COL, ID_COL, md5, readLabels}
import megabyte.communities.util.IO

import scala.collection.JavaConversions._

object SubscribersSelector {

  private val gender = "male"
  private val portalsFileName = s"${gender}Portals.txt"
  private val subscribersFileName = s"${gender}Subscribers.txt"

  def main(args: Array[String]): Unit = {
    val portalIdsFile = new File(portalsDir, portalsFileName)
    val portalIds = IO.readLines(portalIdsFile).map(_.toLong).toSet
    val dao = new MongoDAO(MongoDAO.CRAWLER_DB)
    val userIds = dao.getUserIds(network)
    val labels: Map[String, String] = readLabels(labelsFile, ID_COL, GENDER_COL)
    val subscribers = userIds
      .filter { id =>
        val optLabel = labels.get(md5(id.toString))
        optLabel.isDefined && (optLabel.get == gender)
      }
      .filter(id =>
        dao.getFriends("twitter", id)
          .exists(portalIds.contains(_)))
      .map(_.toString)
    val outFile = new File(portalsDir, subscribersFileName)
    IO.writeLines(subscribers, outFile)
  }
}
