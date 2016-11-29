package megabyte.communities.examples.widget

import java.awt.event.{MouseAdapter, MouseEvent}
import javax.swing.SwingUtilities

class MouseListener(val pointsPane: PointsPane) extends MouseAdapter {

  private val MIN_SELECT_DIST = 10

  override def mousePressed(e: MouseEvent): Unit = {
    val point = getSelectedPoint((e.getX, e.getY))
    if (point.isDefined) {
      if (SwingUtilities.isLeftMouseButton(e)) {
        pointsPane.mlStart match {
          case Some(startPoint) =>
            pointsPane.mlConstraints += ((startPoint, point.get))
            pointsPane.mlStart = None
            pointsPane.notifyListeners()
          case None =>
            if (pointsPane.clStart.isEmpty) {
              pointsPane.mlStart = point
            } else {
              pointsPane.clStart = None
            }
        }
      } else if (SwingUtilities.isRightMouseButton(e)) {
        pointsPane.clStart match {
          case Some(startPoint) =>
            pointsPane.clConstraints += ((startPoint, point.get))
            pointsPane.clStart = None
            pointsPane.notifyListeners()
          case None =>
            if (pointsPane.mlStart.isEmpty) {
              pointsPane.clStart = point
            } else {
              pointsPane.mlStart = None
            }
        }
      }
    } else {
      pointsPane.mlStart = None
      pointsPane.clStart = None
    }
    pointsPane.repaint()
  }

  override def mouseMoved(e: MouseEvent): Unit = {
    pointsPane.curPoint = Some((e.getX, e.getY))
    pointsPane.repaint()
  }

  private def getSelectedPoint(position: (Int, Int)): Option[Int] = {
    val points = pointsPane.points
    val found = points.zipWithIndex.map { case ((x, y), i) =>
      val windowPoint = pointsPane.getCoordinates(x, y)
      val dist = distance(windowPoint, position)
      (i, dist)
    }.minBy(_._2)
    if (found._2 <= MIN_SELECT_DIST) {
      Some(found._1)
    } else {
      None
    }
  }

  private def distance(p1: (Int, Int), p2: (Int, Int)): Double = {
    math.sqrt(math.pow(p1._1 - p2._1, 2) + math.pow(p1._2 - p2._2, 2))
  }
}
