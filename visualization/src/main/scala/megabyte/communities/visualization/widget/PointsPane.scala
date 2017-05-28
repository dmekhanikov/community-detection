package megabyte.communities.visualization.widget

import java.awt.{BasicStroke, Color, Graphics, Graphics2D}
import javax.swing.{JFrame, JPanel}

import scala.collection.mutable.ArrayBuffer

class PointsPane(var points: Seq[(Double, Double)]) extends JPanel {

  private val frame: JFrame = new JFrame("Points")
  private var edges = getEdges(points)
  private var margin = 40
  private var pointSize = 5
  private val callbacks = new ArrayBuffer[() => Unit]()

  private var _clustering : Option[Seq[Int]] = None
  val mlConstraints: ArrayBuffer[(Int, Int)] = new ArrayBuffer()
  val clConstraints: ArrayBuffer[(Int, Int)] = new ArrayBuffer()
  var mlStart: Option[Int] = None
  var clStart: Option[Int] = None
  var curPoint: Option[(Int, Int)] = None

  frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE)
  frame.add(this)
  frame.setSize(800, 600)
  frame.setLocationRelativeTo(null)
  frame.setVisible(true)

  private val _mouseListener = new MouseListener(this)
  addMouseListener(_mouseListener)
  addMouseMotionListener(_mouseListener)

  def setPoints(points: Seq[(Double, Double)]): Unit = {
    this.points = points
    this.edges = getEdges(points)
  }

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
    drawLines(g2d, mlConstraints, Color.MAGENTA)
    drawCurLine(g2d, mlStart, Color.MAGENTA)
    drawLines(g2d, clConstraints, Color.RED)
    drawCurLine(g2d, clStart, Color.RED)
  }

  def drawCurLine(g2d: Graphics2D, start: Option[Int], color: Color) = {
    g2d.setColor(color)
    if (start.isDefined && curPoint.isDefined) {
      val p1 = points(start.get)
      val p1i = getCoordinates(p1._1, p1._2)
      val p2i = curPoint.get
      drawLine(p1i, p2i, g2d)
    }
  }

  def drawLines(g2d: Graphics2D, lines: ArrayBuffer[(Int, Int)], color: Color) = {
    g2d.setColor(color)
    g2d.setStroke(new BasicStroke(2))
    for ((start, end) <- lines) {
      drawLine(points(start), points(end), g2d)
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

  def subscribe(callback: () => Unit): Unit = {
    callbacks += callback
  }

  private[widget] def notifyListeners(): Unit = {
    callbacks.foreach(c => c())
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
