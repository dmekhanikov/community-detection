package megabyte.communities.experiments.util

import com.typesafe.scalalogging.Logger
import megabyte.communities.experiments.config.ExperimentConfig.config._
import megabyte.communities.experiments.dao.MongoDAO
import megabyte.communities.experiments.util.DataUtil.{GENDER_COL, ID_COL, md5, readLabels}

import scala.collection.JavaConversions._
import scala.collection.mutable

object GraphStatistics {

  private val LOG = Logger[GraphStatistics.type]
  private val mongoDAO = new MongoDAO(MongoDAO.CRAWLER_DB)

  def main(args: Array[String]): Unit = {
    val labels: Map[String, String] = readLabels(labelsFile, ID_COL, GENDER_COL)
    val maleSubscribers = mutable.Map[Long, Int]()
    val femaleSubscribers = mutable.Map[Long, Int]()
    val ids = mongoDAO.getUserIds(network)
    for (src <- ids; dest <- mongoDAO.getFriends(network, src)) {
      val srcHash = md5(src.toString)
      if (labels.contains(srcHash)) {
        if (labels(srcHash) == "male") {
          maleSubscribers(dest) = maleSubscribers.getOrElse(dest, 0) + 1
        } else {
          femaleSubscribers(dest) = femaleSubscribers.getOrElse(dest, 0) + 1
        }
      }
    }
    val sortedAccs = (maleSubscribers.keySet ++ femaleSubscribers.keySet).toSeq
      .sortBy { id =>
        maleSubscribers.getOrElse(id, 0) + femaleSubscribers.getOrElse(id, 0)
      }
    sortedAccs.foreach { id =>
      val men = maleSubscribers.getOrElse(id, 0)
      val women = femaleSubscribers.getOrElse(id, 0)
      LOG.info(s"id: $id; men: $men; women: $women")
    }
  }
}
