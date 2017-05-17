package megabyte.communities.experiments.transformer

import java.io.File

import megabyte.communities.experiments.config.ExperimentConfig.config._
import megabyte.communities.experiments.dao.MongoDAO
import megabyte.communities.util.IO

import scala.collection.JavaConversions._

object SubscribersSelector {

  val portalsFileName = "malePortals.txt"
  val subscribersFileName = "maleSubscribers.txt"

  def main(args: Array[String]): Unit = {
    val portalIdsFile = new File(portalsDir, portalsFileName)
    val portalIds = IO.readLines(portalIdsFile).map(_.toLong).toSet
    val dao = new MongoDAO(MongoDAO.CRAWLER_DB)
    val userIds = dao.getUserIds(network)
    val subscribers = userIds
      .filter(id =>
        dao.getFriends("twitter", id)
          .exists(portalIds.contains(_)))
      .map(_.toString)
    val outFile = new File(portalsDir, subscribersFileName)
    IO.writeLines(subscribers, outFile)
  }
}
