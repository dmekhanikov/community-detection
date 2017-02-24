package megabyte.communities.experiments.clustering

import java.io.File
import java.math.BigInteger
import java.security.MessageDigest

import com.typesafe.scalalogging.Logger
import edu.uci.ics.jung.graph.Graph
import edu.uci.ics.jung.graph.util.EdgeType
import megabyte.communities.algo.graph.MultilayerConstrainedSpectralClustering
import megabyte.communities.entities.Edge
import megabyte.communities.experiments.clustering.ClusteringUtil._
import megabyte.communities.experiments.config.ExperimentConfig
import megabyte.communities.util.GraphFactory
import megabyte.communities.util.Graphs._
import org.jblas.DoubleMatrix

import scala.collection.JavaConversions._

private class ConstrainedMultilayerClustering

object ConstrainedMultilayerClustering {

  private val LOG = Logger[ConstrainedMultilayerClustering]

  private val BASE_DIR = ExperimentConfig.config.baseDir
  private val CITY = ExperimentConfig.config.city
  private val GRAPHS_DIR = new File(s"$BASE_DIR/$CITY/graphs/similarity")
  private val CONSTRAINTS_DIR = new File(s"$BASE_DIR/$CITY/graphs/connections")

  private val GRAPH_FILES = Seq(
    "foursquare.csv",
    "twitter.csv",
    "instagram.csv")
  private val CONSTRAINT_FILES = Seq(
    "foursquare.graphml",
    "twitter.graphml",
    "instagram.graphml"
  )

  def main(args: Array[String]): Unit = {
    LOG.info("Reading adjacency matrices")
    val (networksHashes, adjs) = GRAPH_FILES.par.map { fileName =>
      readDataFile(new File(GRAPHS_DIR, fileName))
    }.seq.unzip
    LOG.info("Reading constraint graphs")
    val constraints = CONSTRAINT_FILES.zip(networksHashes).par.map { case (fileName, hashes) =>
      val constraintsFile = new File(CONSTRAINTS_DIR, fileName)
      val graph = GraphFactory.readGraph(constraintsFile)
      val q = adjMatrix(graph, hashes)
      LOG.info("constrains matrix is successfully calculated. Degrees: " +
        degrees(q).map(_.toInt).sortBy(-_).mkString(" "))
      q
    }.seq
    LOG.info("Running multilayer constrained clustering")
    val clustering = MultilayerConstrainedSpectralClustering.getClustering(adjs, constraints, 30, 0.2)
    logStats(clustering)
  }

  private def adjMatrix(graph: Graph[String, Edge], hashes: Seq[String]): DoubleMatrix = {
    val n = hashes.size
    val vertices = graph.getVertices.toSeq
    val positions = vertices.map { v => // vertex id in graph -> position in matrix
      (v, hashes.indexOf(md5(v)))
    }.filter(_._2 > 0)
      .toMap
    val matrix = new DoubleMatrix(n, n)
    for (e <- graph.getEdges) {
      val endpoints = graph.getEndpoints(e)
      val (from, to) = (endpoints.getFirst, endpoints.getSecond)
      for {
        fromInd <- positions.get(from)
        toInd   <- positions.get(to)
      } yield {
        matrix.put(fromInd, toInd, 1)
        if (graph.getEdgeType(e) == EdgeType.UNDIRECTED) {
          matrix.put(toInd, fromInd, 1)
        }
      }
    }
    matrix
  }

  private def md5(s: String): String = {
    val md5Digest = MessageDigest.getInstance("MD5")
    val bytesResult = md5Digest.digest(s.getBytes)
    new BigInteger(1, bytesResult).toString(16)
  }

  private def logStats(clustering: Seq[Int]): Unit = {
    val invClustering = clustering.groupBy(identity)
    val clustersNum = clustering.max + 1
    LOG.info(s"number of clusters: $clustersNum")
    LOG.info(s"sizes:" + invClustering.foldLeft("") { (s, cluster) => s"$s ${cluster._2.size}" })
  }
}
