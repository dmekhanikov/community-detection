package megabyte.communities.visualization.widget

import java.awt.event.{KeyEvent, KeyListener}

class KeyboardListener(private val spaceCallback: () => Unit) extends KeyListener {
  override def keyPressed(e: KeyEvent): Unit = {
    if (e.getKeyCode == KeyEvent.VK_SPACE) {
      spaceCallback()
    }
  }

  override def keyTyped(e: KeyEvent): Unit = {}

  override def keyReleased(e: KeyEvent): Unit = {}
}
