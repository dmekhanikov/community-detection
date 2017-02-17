package megabyte.communities.algo.graph

import com.typesafe.scalalogging.Logger
import edu.uci.ics.jung.graph.Graph
import megabyte.communities.algo.points.XMeans
import megabyte.communities.entities.Edge
import megabyte.communities.util.DoubleMatrixOps._
import megabyte.communities.util.Graphs._
import org.jblas.{DoubleMatrix, Eigen}

private class MultilayerSpectralClustering

object MultilayerSpectralClustering {

  private val LOG = Logger[MultilayerSpectralClustering]

  // alpha - coefficient in the objective function. Importance of layers to be close on the manifold.
  def getClustering(graphs: Seq[Graph[Long, Edge]], k: Int, alpha: Double): Map[Long, Int] = {
    val numeration = numerateNodes(graphs.head)
    val adjMatrices = graphs.par.map { g =>
      val renumG = applyNumeration(g, numeration)
      adjacencyMatrix(renumG)
    }
    val renumClustering = getClustering(adjMatrices.seq, k, alpha)
    Map(numeration.zip(renumClustering): _*)
  }

  def getClustering(adjMatrices: Seq[DoubleMatrix], k: Int, alpha: Double): Seq[Int] = {
    LOG.info(s"Starting multilayer clustering computation")
    val lSyms = adjMatrices.map(symLaplacian)
    val us = lSyms.zipWithIndex.map { case (l, i) =>
      LOG.info(s"Starting processing layer #${i + 1}")
      val u = toEigenspace(l, k)
      LOG.info(s"${i + 1} / ${adjMatrices.size} layers processed")
      u
    }
    LOG.info("Building modified Laplacian matrix")
    val n = adjMatrices.head.rows
    val lMod = new DoubleMatrix(n, n)
    lSyms.zip(us).foreach { case (li, ui) =>
      lMod += (li -= ((ui * ui.transpose()) *= alpha))
    }
    LOG.info("Running spectral clustering on the modified Laplacian matrix")
    val u = toEigenspace(lMod, k).normRowsI()
    XMeans.getClustering(u)
  }

  def toEigenspace(matrix: DoubleMatrix, dim: Int): DoubleMatrix = {
    val Array(vectors, values) = Eigen.symmetricEigenvectors(matrix)
    // sort by values and take first k
    val indices = (0 until values.columns)
      .map(i => matrix.get(i, i))
      .zipWithIndex
      .sortBy(_._1)
      .take(dim)
      .map(_._2)
      .toArray
    vectors.getColumns(indices)
  }
}
