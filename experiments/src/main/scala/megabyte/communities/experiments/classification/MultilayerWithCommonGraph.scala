package megabyte.communities.experiments.classification

import java.io.File

import megabyte.communities.experiments.config.ExperimentConfig.config._
import megabyte.communities.experiments.util.DataUtil._
import megabyte.communities.util.DoubleMatrixOps._
import megabyte.communities.util.Graphs

object MultilayerWithCommonGraph {

  def main(args: Array[String]): Unit = {
    val relationFile = new File(relationsDir, "multilayer-common-graph.csv")

    val (networksHashes, adjs) = networks.par.map(readAdj).seq.unzip
    val dataLSyms = adjs.map(Graphs.symLaplacian)
    val dataUs = networks.zip(dataLSyms).map {
      case (net, l) => readOrCalcSymSubspace(net, l)
    }

    val qs = networks.zip(networksHashes).par.map { case (net, hashes) =>
      readConstraintsMatrix(s"$net.graphml", hashes)
    }.seq
    val sumQ = qs.reduce((a, b) => a += b)
    val qLSym = Graphs.symLaplacian(sumQ)
    val qU = readOrCalcSymSubspace("common-graph", qLSym)

    MultilayerSpectral.run(dataLSyms :+ sumQ, dataUs :+ qU, relationFile)
  }
}
