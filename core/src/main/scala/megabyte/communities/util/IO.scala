package megabyte.communities.util

import java.io.File

import com.typesafe.scalalogging.Logger
import org.jblas.DoubleMatrix
import DoubleMatrixOps._

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
}
