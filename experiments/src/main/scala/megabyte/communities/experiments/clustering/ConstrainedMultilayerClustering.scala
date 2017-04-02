package megabyte.communities.experiments.clustering

import java.io.File
import java.math.BigInteger
import java.security.MessageDigest

import com.typesafe.scalalogging.Logger
import edu.uci.ics.jung.graph.Graph
import edu.uci.ics.jung.graph.util.EdgeType
import megabyte.communities.algo.graph.{ConstrainedSpectralClustering, MultilayerConstrainedSpectralClustering}
import megabyte.communities.algo.points.KMeans
import megabyte.communities.entities.Edge
import megabyte.communities.experiments.config.ExperimentConfig.config._
import megabyte.communities.util.DoubleMatrixOps._
import megabyte.communities.util.GraphFactory
import megabyte.communities.util.IO._
import org.jblas.DoubleMatrix

import scala.collection.JavaConversions._

private class ConstrainedMultilayerClustering

object ConstrainedMultilayerClustering {

  private val LOG = Logger[ConstrainedMultilayerClustering]

  private val k = 2
  private val alpha = 0.2

  def main(args: Array[String]): Unit = {
    LOG.info("Reading adjacency matrices")
    val (networksHashes, adjs) = networks.par.map { network =>
      readMatrixWithHeader(new File(similarityGraphsDir, s"$network.csv"))
    }.seq.unzip
    LOG.info("Reading constraint graphs")
    val constraints = networks.zip(networksHashes).par.map { case (net, hashes) =>
      readConstraintsMatrix(s"$net.graphml", hashes)
    }.seq

    LOG.info("Calculating subspace representations for each layer with applied constraints")
    val us = networks.zip(adjs).zip(constraints).par.map { case ((net, adj), q) =>
      val file = new File(subspaceDir, s"$net.csv")
      readOrCalcMatrix(file) {
        ConstrainedSpectralClustering.toEigenspace(adj, q)
      }.prefixColumns(k)
    }.seq
    val u = MultilayerConstrainedSpectralClustering.combineLayers(adjs, us, k, alpha)

    val clustering = KMeans.getClustering(u, k)
    logStats(clustering)
  }

  private def readConstraintsMatrix(fileName: String, hashes: Seq[String]): DoubleMatrix = {
    val constraintsFile = new File(socialGraphsDir, fileName)
    val graph = GraphFactory.readGraph(constraintsFile)
    val q = adjMatrix(graph, hashes)
    q
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
