from PySide6.QtGui import QSurfaceFormat, QVector3D
import numpy as np
from glaxisitem import GLAxisItem

from glgriditem import GLGridItem
from glwidget import GLWidget


class RenderWidget(GLWidget):
    def __init__(self):
        super().__init__()
        self.grid = GLGridItem(size=(7, 7), lineWidth=1)
        self.ax = GLAxisItem(size=np.array([8, 8, 8]))
        self.addItem(self.grid)
        self.addItem(self.ax)
