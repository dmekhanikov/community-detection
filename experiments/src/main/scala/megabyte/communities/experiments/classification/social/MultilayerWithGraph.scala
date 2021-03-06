package megabyte.communities.experiments.classification.social

import megabyte.communities.experiments.config.ExperimentConfig.config._
import megabyte.communities.experiments.util.DataUtil._
import megabyte.communities.util.Graphs

object MultilayerWithGraph {

  def main(args: Array[String]): Unit = {
    val (networksHashes, adjs) = networks.par.map(readAdj).seq.unzip
    val dataLSyms = adjs.map(Graphs.symLaplacian)
    val dataUs = networks.zip(dataLSyms).map {
      case (net, l) => readOrCalcSymSubspace(net, l)
    }

    val qs = networks.zip(networksHashes).par.map { case (net, hashes) =>
      readConstraintsMatrix(s"$net.graphml", hashes)
    }.seq
    val qLSyms = qs.map(Graphs.symLaplacian)
    val qUs = networks.zip(dataLSyms).map {
      case (net, l) => readOrCalcSymSubspace(net + "-graph", l)
    }

    MultilayerSpectral.run(dataLSyms ++ qLSyms, dataUs ++ qUs)
  }
}
