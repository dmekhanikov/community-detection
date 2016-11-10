package megabyte.communities.examples

import java.awt.{Color, Graphics, Graphics2D}
import javax.swing.{JFrame, JPanel}

class PointsPane(val points: Seq[(Double, Double, Color)]) extends JPanel {

  private val frame: JFrame = new JFrame("Points")
  private val edges = getEdges(points)
  var margin = 30
  var pointSize = 5

  frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE)
  frame.add(this)
  frame.setSize(800, 600)
  frame.setLocationRelativeTo(null)
  frame.setVisible(true)

  override def paintComponent(g: Graphics): Unit = {
    super.paintComponent(g)

    val g2d = g.asInstanceOf[Graphics2D]
    points.foreach { case (x, y, color) =>
      g2d.setColor(color)
      val (intX, intY) = getCoordinates(x, y, edges)
      g2d.fillOval(intX, intY, pointSize, pointSize)
    }
  }

  private def getCoordinates(x: Double, y: Double, edges: PaneEdges): (Int, Int) = {
    val w = frame.getContentPane.getWidth
    val h = frame.getContentPane.getHeight
    (getCoordinate(x, edges.xMin, edges.xMax, w),
      h - getCoordinate(y, edges.yMin, edges.yMax, h))
  }

  private def getCoordinate(c: Double, cMin: Double, cMax: Double, newSize: Int): Int = {
    margin + ((c - cMin) / (cMax - cMin) * (newSize - 2 * margin)).toInt
  }

  private def getEdges(points: Seq[(Double, Double, Color)]): PaneEdges = {
    val xs = points.map(_._1)
    val ys = points.map(_._2)
    new PaneEdges(xs.min, xs.max, ys.min, ys.max)
  }

  class PaneEdges(val xMin: Double,
                  val xMax: Double,
                  val yMin: Double,
                  val yMax: Double)

}
