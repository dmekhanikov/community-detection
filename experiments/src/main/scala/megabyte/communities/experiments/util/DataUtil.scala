package megabyte.communities.experiments.util

import java.io.File

import com.typesafe.scalalogging.Logger
import megabyte.communities.util.IO
import megabyte.communities.util.IO.readCSVToSeq
import org.jblas.DoubleMatrix
import weka.filters.unsupervised.attribute.Remove

import scala.collection.TraversableLike
import scala.collection.mutable.ArrayBuffer

private class DataUtil

object DataUtil {

  type Features = Seq[Double]
  type Users = Map[String, Features]

  private val LOG = Logger[DataUtil]

  val ID_COL = "row ID"
  val GENDER_COL = "gender"
  val GENDER_VALUES = Seq("male", "female")

  def split[T, S <: Seq[T]](s: TraversableLike[T, S], leftFraction: Double): (S, S) = {
    val leftSize = (s.size * leftFraction).toInt
    s.splitAt(leftSize)
  }

  def readLabels(file: File, idCol: String, labelCol: String): Map[String, String] = {
    IO.readCSVToMap(file)
      .map(m => m(idCol) -> m(labelCol))
      .filter { case (_, v) => v.nonEmpty }
      .toMap
  }

  def getLabels[T, E](indices: Seq[Int], numeration: Seq[T], labels: Map[T, E]): Seq[E] = {
    indices
      .map(i => numeration(i))
      .map(id => labels(id))
  }

  def readUsersData(file: File): Users = {
    LOG.info(s"Reading users data from file $file")
    val records = readCSVToSeq(file)
    records.map { record =>
      val id = record.head
      val features = record.tail.map(_.toDouble)
      id -> features
    }.toMap
  }

  def foldFeatures(values: Iterable[Features], f: (Double, Double) => Double): Features = {
    values.tail.fold(values.head) { (min: Features, cur: Features) =>
      min.zip(cur).map { case (a, b) => f(a, b) }
    }
  }

  def normalizeFeatures(users: Users): Users = {
    val values = users.values
    val mins = foldFeatures(values, math.min)
    val maxs = foldFeatures(values, math.max)
    users.map { case (userId, features) =>
      (userId,
        features.zip(mins.zip(maxs))
          .map { case (value, (min, max)) =>
            if (min != max) {
              (value - min) / (max - min)
            } else {
              value
            }
          })
    }
  }

  def mergeFeatures(dataSets: Seq[Users]): Users = {
    val result = dataSets.head.map { case (k, v) =>
      val ab = new ArrayBuffer[Double]()
      ab ++= v
      (k, ab)
    }
    dataSets.tail.foreach { dataSet =>
      dataSet.foreach { case (k, f) =>
        result(k) ++= f
      }
    }
    result
  }

  def findUsersIntersection(userDataSets: Seq[Users]): Set[String] = {
    val userSets = userDataSets.map {
      _.keys.toSet
    }
    userSets.tail.foldLeft(userSets.head) { case (a, b) => a.intersect(b) }
  }

  def mergeUserData(networksData: Map[String, Seq[Users]]): Map[String, Users] = {
    val usersIntersection = findUsersIntersection(networksData.values.flatten.toSeq)
    LOG.info("Merging features")
    networksData.mapValues { (networkUsers: Seq[Users]) =>
      val unifiedUsers = networkUsers.map { (users: Users) =>
        users.filter { case (k, _) => usersIntersection.contains(k) }
      }
      mergeFeatures(unifiedUsers)
    }
  }

  def makeFeaturesMatrix(users: Users, ids: Seq[String]): DoubleMatrix = {
    val features = ids.map(id => users(id).toArray)
    new DoubleMatrix(features.toArray)
  }

  def concatFeatures(networkUsers: Map[String, Users]): Users = {
    networkUsers.values.reduce { (u1, u2) =>
      val ids = u1.keys
      ids.map { id =>
        id -> (u1(id) ++ u2(id))
      }.toMap
    }
  }

  def attributesPrefixFilter(prefixLen: Int): Remove = {
    val remove = new Remove
    remove.setAttributeIndices(s"first-$prefixLen,last")
    remove.setInvertSelection(true)
    remove
  }
}
