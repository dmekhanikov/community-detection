package megabyte.communities.experiments.util

import java.io.File
import java.math.BigInteger
import java.security.MessageDigest

import com.typesafe.scalalogging.Logger
import edu.uci.ics.jung.graph.Graph
import edu.uci.ics.jung.graph.util.EdgeType
import megabyte.communities.algo.graph.{LuConstrainedSpectralClustering, SpectralClustering}
import megabyte.communities.entities.Edge
import megabyte.communities.experiments.config.ExperimentConfig.config._
import megabyte.communities.util.IO.{readCSVToSeq, readMatrixWithHeader, readOrCalcMatrix}
import megabyte.communities.util.{GraphFactory, IO}
import org.jblas.DoubleMatrix
import weka.filters.unsupervised.attribute.Remove

import scala.collection.JavaConversions._
import scala.collection.TraversableLike
import scala.collection.mutable.ArrayBuffer

object DataUtil {

  type Features = Seq[Double]
  type Users = Map[String, Features]

  private val LOG = Logger[DataUtil.type]

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

  def readAdj(network: String): (Seq[String], DoubleMatrix) = {
    readMatrixWithHeader(new File(similarityGraphsDir, network + ".csv"))
  }

  def readOrCalcSymSubspace(net: String, lSym: DoubleMatrix): DoubleMatrix = {
    val file = new File(symSubspacesDir, net + ".csv")
    readOrCalcMatrix(file) {
      SpectralClustering.toEigenspace(lSym)
    }
  }

  def readIds(net: String): Seq[String] = {
    IO.readHeader(new File(similarityGraphsDir, net + ".csv"))
  }

  def readConstraintsMatrix(fileName: String, hashes: Seq[String]): DoubleMatrix = {
    val constraintsFile = new File(socialGraphsDir, fileName)
    val graph = GraphFactory.readGraph(constraintsFile)
    val q = graphToAdjMatrix(graph, hashes)
    q
  }

  def graphToAdjMatrix(graph: Graph[String, Edge], numeration: Seq[String]): DoubleMatrix = {
    val n = numeration.size
    val vertices = graph.getVertices.toSeq
    val positions = vertices.map { v => // vertex id in graph -> position in matrix
      (v, numeration.indexOf(md5(v)))
    }.filter(_._2 > 0)
      .toMap
    val matrix = new DoubleMatrix(n, n)
    for (e <- graph.getEdges) {
      val endpoints = graph.getEndpoints(e)
      val (from, to) = (endpoints.getFirst, endpoints.getSecond)
      for {
        fromInd <- positions.get(from)
        toInd <- positions.get(to)
      } yield {
        matrix.put(fromInd, toInd, 1)
        if (graph.getEdgeType(e) == EdgeType.UNDIRECTED) {
          matrix.put(toInd, fromInd, 1)
        }
      }
    }
    matrix
  }

  def md5(s: String): String = {
    val md5Digest = MessageDigest.getInstance("MD5")
    val bytesResult = md5Digest.digest(s.getBytes)
    new BigInteger(1, bytesResult).toString(16)
  }
}
