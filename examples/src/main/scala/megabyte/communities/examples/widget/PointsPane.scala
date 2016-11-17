package megabyte.communities.examples.widget

import java.awt.{BasicStroke, Color, Graphics, Graphics2D}
import javax.swing.{JFrame, JPanel}

import scala.collection.mutable.ArrayBuffer

class PointsPane(val points: Seq[(Double, Double)]) extends JPanel {

  private val frame: JFrame = new JFrame("Points")
  private val edges = getEdges(points)
  private var margin = 30
  private var pointSize = 5

  private var _clustering : Option[Seq[Int]] = None
  val constraints: ArrayBuffer[(Int, Int)] = new ArrayBuffer()
  var selectedPoint: Option[Int] = None
  var curPoint: Option[(Int, Int)] = None

  frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE)
  frame.add(this)
  frame.setSize(800, 600)
  frame.setLocationRelativeTo(null)
  frame.setVisible(true)

  private val _mouseListener = new MouseListener(this)
  addMouseListener(_mouseListener)
  addMouseMotionListener(_mouseListener)

  override def paintComponent(g: Graphics): Unit = {
    super.paintComponent(g)

    val g2d = g.asInstanceOf[Graphics2D]

    // draw points
    for (i <- points.indices) {
      val (x, y) = points(i)
      val color = clustering match {
        case Some(clust) => getColor(clust(i))
        case None => Color.BLACK
      }
      g2d.setColor(color)
      val (intX, intY) = getCoordinates(x, y)
      g2d.fillOval(intX, intY, pointSize, pointSize)
    }

    // draw constraints
    g2d.setColor(Color.BLUE)
    g2d.setStroke(new BasicStroke(2))
    for ((start, end) <- constraints) {
      drawLine(points(start), points(end), g2d)
    }
    if (selectedPoint.isDefined && curPoint.isDefined) {
      val p1 = points(selectedPoint.get)
      val p1i = getCoordinates(p1._1, p1._2)
      val p2i = curPoint.get
      drawLine(p1i, p2i, g2d)
    }
  }

  private def drawLine(p1: (Double, Double), p2: (Double, Double), g2d: Graphics2D): Unit = {
    val p1i = getCoordinates(p1._1, p1._2)
    val p2i = getCoordinates(p2._1, p2._2)
    drawLine(p1i, p2i, g2d)
  }

  private def drawLine(p1: (Int, Int), p2: (Int, Int), g2d: Graphics2D): Unit = {
    val dx = pointSize / 2
    val dy = pointSize / 2
    g2d.drawLine(p1._1 + dx, p1._2 + dy, p2._1 + dx, p2._2 + dy)
  }

  def clustering = _clustering

  def clustering_=(clustering: Seq[Int]): Unit = {
    this._clustering = Some(clustering)
    repaint()
  }

  private def getColor(i: Int): Color = {
    val colors = Seq(
      Color.RED,
      Color.BLUE,
      Color.BLACK,
      Color.MAGENTA,
      Color.CYAN,
      Color.ORANGE,
      Color.GRAY
    )
    colors(math.min(i, colors.size - 1))
  }

  def getCoordinates(x: Double, y: Double): (Int, Int) = {
    val w = frame.getContentPane.getWidth
    val h = frame.getContentPane.getHeight
    (getCoordinate(x, edges.xMin, edges.xMax, w),
      h - getCoordinate(y, edges.yMin, edges.yMax, h))
  }

  private def getCoordinate(c: Double, cMin: Double, cMax: Double, newSize: Int): Int = {
    margin + ((c - cMin) / (cMax - cMin) * (newSize - 2 * margin)).toInt
  }

  private def getEdges(points: Seq[(Double, Double)]): PaneEdges = {
    val xs = points.map(_._1)
    val ys = points.map(_._2)
    new PaneEdges(xs.min, xs.max, ys.min, ys.max)
  }

  class PaneEdges(val xMin: Double,
                  val xMax: Double,
                  val yMin: Double,
                  val yMax: Double)

}
