package megabyte.communities.experiments

import java.io.File

import com.typesafe.scalalogging.Logger
import megabyte.communities.algo.graph.SpectralClustering
import megabyte.communities.util.IO
import megabyte.communities.util.Measures.modularity
import org.jblas.DoubleMatrix

import scala.collection.TraversableLike

private class Util

object Util {

  private val LOG = Logger[Util]

  val ID_COL = "row ID"
  val GENDER_COL = "gender"
  val GENDER_VALUES = Seq("male", "female")

  def optimizeClustersCount(adj: DoubleMatrix, start: Int, end: Int): (Int, Seq[Int]) = {
    (start to end)
      .map { k =>
        val clustering = SpectralClustering.getClustering(adj, k)
        logStats(adj, k, clustering)
        (k, clustering)
      }.maxBy { case (_, clustering) => modularity(adj, clustering) }
  }

  def logStats(adj: DoubleMatrix, k: Int, clustering: Seq[Int]): Unit = {
    val modul = modularity(adj, clustering)
    val invClustering = clustering.groupBy(identity)
    val clustersNum = clustering.max + 1
    LOG.info(s"subspace dimensionality: $k")
    LOG.info(s"number of clusters: $clustersNum")
    LOG.info(s"sizes: " + (invClustering.map { cluster => cluster._2.size }.toSeq.sorted.reverse mkString " "))
    LOG.info(s"modularity: $modul")
  }

  def split[T, S <: Seq[T]](s: TraversableLike[T, S], leftFraction: Double): (S, S) = {
    val leftSize = (s.size * leftFraction).toInt
    s.splitAt(leftSize)
  }

  def readLabels(file: File, idCol: String, labelCol: String): Map[String, String] = {
    IO.readCSV(file)
      .map(m => m(idCol) -> m(labelCol))
      .filter { case (_, v) => v.nonEmpty }
      .toMap
  }

  def getLabels[T, E](indices: Seq[Int], numeration: Seq[T], labels: Map[T, E]): Seq[E] = {
    indices
      .map(i => numeration(i))
      .map(id => labels(id))
  }
}
