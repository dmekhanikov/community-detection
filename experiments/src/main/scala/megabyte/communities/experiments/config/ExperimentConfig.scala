package megabyte.communities.experiments.config

import java.io.{BufferedReader, File, FileNotFoundException, FileReader}
import java.util.Properties

import scalaz.Scalaz._

class ExperimentConfig private(val baseDir: String, val city: String, val network: Option[String])

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
    val network = Option(properties.getProperty(NETWORK_PROP))
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
