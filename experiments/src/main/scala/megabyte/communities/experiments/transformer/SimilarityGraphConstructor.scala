package megabyte.communities.experiments.transformer

import java.io.{BufferedWriter, File, FileWriter}

import com.typesafe.scalalogging.Logger
import megabyte.communities.experiments.config.ExperimentConfig
import megabyte.communities.util.Measures
import org.jblas.DoubleMatrix

import scala.collection.mutable.ArrayBuffer

private class SimilarityGraphConstructor

object SimilarityGraphConstructor {

  private val LOG = Logger[SimilarityGraphConstructor]

  private val BASE_DIR = ExperimentConfig.config.baseDir
  private val CITY = ExperimentConfig.config.city
  private val FEATURES_DIR = new File(s"$BASE_DIR/$CITY/features")
  private val GRAPHS_DIR = new File(s"$BASE_DIR/$CITY/graphs/similarity")
  private val NETWORKS = Seq(
    ("twitter", Seq("LDA50Features.csv", "LIWCFeatures.csv", "manuallyDefinedTextFeatures.csv")),
    ("instagram", Seq("imageConceptsFeatures.csv")),
    ("foursquare", Seq("venueCategoriesFeatures5Months.csv"))
  )

  type Features = Seq[Double]
  type Users = Map[String, Features]

  def main(args: Array[String]): Unit = {
    val networksData: Seq[(String, Seq[Users])] =
      NETWORKS.map { case (networkName, files) =>
        (networkName, files.map { file => readDataFile(new File(FEATURES_DIR, s"$networkName/$file")) })
      }
    val usersIntersection = findUsersIntersection(networksData.flatMap(_._2))
    LOG.info("Merging features")
    val mergedData: Seq[(String, Users)] =
      map2(networksData) { (networkUsers: Seq[Users]) =>
        val unifiedUsers = networkUsers.map { (users: Users) =>
          users.filter { case (k, _) => usersIntersection.contains(k) }
        }
        mergeFeatures(unifiedUsers)
      }
    val normalizedData: Seq[(String, Users)] = map2(mergedData) {
      normalizeFeatures
    }
    val numeration = usersIntersection.toSeq
    normalizedData.par.foreach { case (network, users) =>
      LOG.info(s"Calculating adjacency matrix for $network")
      val adj = calcAdjMatrix(users, numeration)
      val outFile = new File(GRAPHS_DIR, s"$network.csv")
      LOG.info(s"Writing result for $network to $outFile")
      writeMatrix(adj, numeration, outFile)
    }
    LOG.info("Finished, exiting")
  }

  private def map2[A, B](fa: Seq[(String, A)])(f: A => B): Seq[(String, B)] = {
    fa.map { case (s, x) => (s, f(x)) }
  }

  private def foldFeatures(values: Iterable[Features], f: (Double, Double) => Double): Features = {
    values.tail.fold(values.head) { (min: Features, cur: Features) =>
      min.zip(cur).map { case (a, b) => f(a, b) }
    }
  }

  private def normalizeFeatures(users: Users): Users = {
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

  private def mergeFeatures(dataSets: Seq[Users]): Users = {
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

  private def calcAdjMatrix(users: Users, numeration: Seq[String]): DoubleMatrix = {
    val n = users.keys.size
    val adj = new DoubleMatrix(n, n)
    for (i <- (0 until n).par; j <- (0 until n).par) {
      val a = users(numeration(i))
      val b = users(numeration(j))
      adj.put(i, j, Measures.cosineSim(a, b))
    }
    adj
  }

  private def findUsersIntersection(userDataSets: Seq[Users]): Set[String] = {
    val userSets = userDataSets.map {
      _.keys.toSet
    }
    userSets.tail.foldLeft(userSets.head) { case (a, b) => a.intersect(b) }
  }

  private def readDataFile(file: File): Users = {
    LOG.info(s"Reading data from file $file")
    val source = io.Source.fromFile(file)
    source.getLines.drop(1).map { line =>
      val tokens = line.split(",").map(_.trim)
      val id = unquote(tokens(0))
      val features = tokens.tail.map(_.toDouble).toSeq
      (id, features)
    }.toMap
  }

  private def unquote(s: String): String = {
    s.filter {c => c != '"' }
  }

  private def writeMatrix(matrix: DoubleMatrix, header: Seq[String], file: File): Unit = {
    val writer = new BufferedWriter(new FileWriter(file))
    writer.write(header.mkString(",") + "\n")
    val adjText = matrix.toString("%f", "", "", ",", "\n")
    writer.write(adjText)
    writer.close()
  }
}
