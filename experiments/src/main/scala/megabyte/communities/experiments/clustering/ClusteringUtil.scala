package megabyte.communities.experiments.clustering

import java.io.File

import com.typesafe.scalalogging.Logger
import megabyte.communities.algo.graph.SpectralClustering
import megabyte.communities.util.DoubleMatrixOps._
import megabyte.communities.util.Measures.modularity
import org.jblas.DoubleMatrix

private class ClusteringUtil

object ClusteringUtil {

  private val LOG = Logger[ClusteringUtil]

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

  def readDataFile(file: File): (Seq[String], DoubleMatrix) = {
    LOG.info(s"Reading data from file $file")
    val source = io.Source.fromFile(file)
    val lines = source.getLines
    val header = lines.next().split(",")
    val data = lines.map { line =>
      line.split(",").map(_.trim).map(_.toDouble).array
    }.toArray
    (header, new DoubleMatrix(data))
  }

  def readMatrix(file: File): DoubleMatrix = {
    val source = io.Source.fromFile(file)
    try {
      val lines = source.getLines
      val data = lines.map { line =>
        line.split(",").map(_.trim).map(_.toDouble).array
      }.toArray
      new DoubleMatrix(data)
    } finally {
      source.close()
    }
  }

  def readOrCalcMatrix(file: File)(calculator: => DoubleMatrix): DoubleMatrix = {
    if (file.exists()) {
      LOG.info(s"File with a matrix found: $file")
      readMatrix(file)
    } else {
      LOG.info(s"File with a matrix not found: $file. Calculating...")
      val m = calculator
      LOG.info(s"Writing a calculated matrix to file: $file")
      file.getParentFile.mkdirs()
      m.write(file, lossless = true)
      m
    }
  }
}
