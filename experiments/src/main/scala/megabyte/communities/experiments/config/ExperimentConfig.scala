package megabyte.communities.experiments.config

import java.io.{BufferedReader, File, FileNotFoundException, FileReader}
import java.util.Properties

import scalaz.Scalaz._

class ExperimentConfig private(val baseDir: String, val city: String, val network: String) {
  val cityDir = new File(baseDir, city)
  val idsDir = new File(cityDir, "ids")
  val trainIdsFile = new File(idsDir, "train.txt")
  val testIdsFile = new File(idsDir, "test.txt")
  val graphsDir = new File(cityDir, "graphs")
  val similarityGraphsDir = new File(graphsDir, "similarity")
  val labelsDir = new File(cityDir, "labels")
  val labelsFile = new File(labelsDir, s"${city}GroundTruth.csv")
  val subspacesDir = new File(cityDir, "subspaces")
  val symSubspacesDir = new File(subspacesDir, "sym")
  val constrainedSubspacesDir = new File(subspacesDir, "constrained")
  val commonConstrainedSubspacesDir = new File(constrainedSubspacesDir, "common")
  val socialGraphsDir = new File(graphsDir, "connections")
  val featuresDir = new File(cityDir, "features")
  val relationsDir = new File(cityDir, "relations")
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
  private val CITY_PROP = "city"
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
    val city = Option(properties.getProperty(CITY_PROP))
      .getOrElse(throwNoProperty(CITY_PROP))
    val network = properties.getProperty(NETWORK_PROP)
    new ExperimentConfig(dataDir, city, network)
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
