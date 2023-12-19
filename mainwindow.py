from PySide6.QtWidgets import QMainWindow
from PySide6.QtGui import QSurfaceFormat

from renderwidget import RenderWidget


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("OpenGL Widget")
        self.resize(800, 600)
        self.format = QSurfaceFormat()
        self.format.setSamples(4)
        self.format.setVersion(3, 3)
        self.format.setProfile(QSurfaceFormat.OpenGLContextProfile.NoProfile)
        self.format.setSwapBehavior(QSurfaceFormat.SwapBehavior.DoubleBuffer)
        self.glwidget = RenderWidget()
        self.setCentralWidget(self.glwidget)

        self.show()
