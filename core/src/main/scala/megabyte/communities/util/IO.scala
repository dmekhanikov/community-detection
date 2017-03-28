package megabyte.communities.util

import java.io.{BufferedReader, File, FileReader}

import com.typesafe.scalalogging.Logger
import megabyte.communities.util.DoubleMatrixOps._
import org.apache.commons.csv.CSVFormat
import org.jblas.DoubleMatrix
import weka.core.Instances
import weka.core.converters.ArffSaver

import scala.collection.JavaConversions._

private class IO

object IO {

  private val LOG = Logger[IO]

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

  def readCSV(file: File): Seq[Map[String, String]] = {
    val reader = new FileReader(file)
    val parser = CSVFormat.DEFAULT.withFirstRecordAsHeader().parse(reader)
    try {
      parser.map(_.toMap.toMap).toSeq // first toMap makes util.Map, second - immutable.Map
    } finally {
      parser.close()
      reader.close()
    }
  }

  def readHeader(file: File): Seq[String] = {
    val source = io.Source.fromFile(file)
    try {
      source.getLines().next().split(',')
    } finally {
      source.close()
    }
  }

  def readInstances(file: File): Instances = {
    val reader = new BufferedReader(new FileReader(file))
    try {
      new Instances(reader)
    } finally {
      reader.close()
    }
  }

  def writeInstances(instances: Instances, file: File): Unit = {
    val saver = new ArffSaver()
    saver.setInstances(instances)
    saver.setFile(file)
    saver.writeBatch()
  }
}
