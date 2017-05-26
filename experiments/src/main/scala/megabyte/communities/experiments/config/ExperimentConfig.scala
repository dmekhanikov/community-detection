package megabyte.communities.experiments.config

import java.io.{BufferedReader, File, FileNotFoundException, FileReader}
import java.util.Properties

import scalaz.Scalaz._

class ExperimentConfig private(val baseDir: String, val dataset: String, val network: String) {
  val datasetDir = new File(baseDir, dataset)
  val idsDir = new File(datasetDir, "ids")
  val allIdsFile = new File(baseDir, "allIds.txt")
  val trainIdsFile = new File(idsDir, "train.txt")
  val testIdsFile = new File(idsDir, "test.txt")
  val graphsDir = new File(datasetDir, "graphs")
  val similarityGraphsDir = new File(graphsDir, "similarity")
  val constrainedGraphsDir = new File(graphsDir, "constrained")
  val labelsDir = new File(datasetDir, "labels")
  val labelsFile = new File(labelsDir, s"${dataset}GroundTruth.csv")
  val subspacesDir = new File(datasetDir, "subspaces")
  val symSubspacesDir = new File(subspacesDir, "sym")
  val constrainedSubspacesDir = new File(subspacesDir, "constrained")
  val commonConstrainedSubspacesDir = new File(constrainedSubspacesDir, "common")
  val socialGraphsDir = new File(graphsDir, "connections")
  val featuresDir = new File(datasetDir, "features")
  val portalsDir = new File(baseDir, "portals")
  val featureFiles: Map[String, Seq[File]] = Map(
    ("twitter", Seq("LDA50Features", "LIWCFeatures", "manuallyDefinedTextFeatures")),
    ("instagram", Seq("imageConceptsFeatures")),
    ("foursquare", Seq("venueCategoriesFeatures5Months"))
  ).map { case (net, fileNames) =>
    net -> fileNames.map {
      fileName => new File(featuresDir, s"$net/$fileName.csv")
    }
  }
  val networks = Seq("twitter", "instagram", "foursquare")
}

object ExperimentConfig {

  private val CONFIG_FILE_NAME = "config.properties"
  private val DATASET_PROP = "dataset"
  private val DATA_DIR_PROP = "dataDir"
  private val NETWORK_PROP = "network"

  lazy val config: ExperimentConfig = load()

  private def load(): ExperimentConfig = {
    val configFile = findConfigFile(CONFIG_FILE_NAME)
      .getOrElse(throw new FileNotFoundException("Config file not found"))
    val configFileReader = new BufferedReader(new FileReader(configFile))
    val properties = new Properties()
    try {
      properties.load(new FileReader(configFile))
    } finally {
      configFileReader.close()
    }
    val dataDir = Option(properties.getProperty(DATA_DIR_PROP))
      .getOrElse(throwNoProperty(DATA_DIR_PROP))
    val dataset = Option(properties.getProperty(DATASET_PROP))
      .getOrElse(throwNoProperty(DATASET_PROP))
    val network = properties.getProperty(NETWORK_PROP)
    new ExperimentConfig(dataDir, dataset, network)
  }

  private def throwNoProperty(prop: String): Nothing = {
    throw new NoSuchElementException(s"""property "$prop" was not found""")
  }

  private def findConfigFile(fileName: String): Option[File] = {
    val localFile = new File(CONFIG_FILE_NAME)
    if (localFile.exists) {
      localFile.some
    } else {
      Option(ExperimentConfig.getClass.getClassLoader.getResource(CONFIG_FILE_NAME)) >>= {
        url => new File(url.getFile).some
      }
    }
  }
}
