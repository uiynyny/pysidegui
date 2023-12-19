import numpy as np
from PySide6.QtGui import QMatrix4x4, QVector3D
from bufferobject import VAO, VBO
from glgraphicsitem import GLGraphicsItem
from OpenGL import GL

from shader import Shader

vertex_shader = """
#version 330 core
uniform mat4 model;
uniform mat4 view;

layout (location = 0) in vec3 iPos;

void main() {
    gl_Position = view * model * vec4(iPos, 1.0);
}
"""

fragment_shader = """
#version 330 core
out vec4 FragColor;

uniform vec4 objColor1;


void main() {
    FragColor = objColor1;
}
"""


def make_grid_data(size, spacing):
    x, y = size
    dx, dy = spacing
    xvals = np.arange(-x / 2.0, x / 2.0 + dx * 0.001, dx, dtype=np.float32)
    yvals = np.arange(-y / 2.0, y / 2.0 + dy * 0.001, dy, dtype=np.float32)

    xlines = np.stack(
        np.meshgrid(xvals, [yvals[0], yvals[-1]], indexing="ij"), axis=2
    ).reshape(-1, 2)
    ylines = np.stack(
        np.meshgrid([xvals[0], xvals[-1]], yvals, indexing="xy"), axis=2
    ).reshape(-1, 2)
    data = np.concatenate([xlines, ylines], axis=0)
    data = np.pad(data, ((0, 0), (0, 1)), mode="constant", constant_values=0)
    return data


class GLGridItem(GLGraphicsItem):
    def __init__(
        self,
        size=(1.0, 1.0),
        spacing=(1.0, 1.0),
        color=(1.0, 1.0, 1.0, 0.4),
        lineWidth=1,
        antialias=True,
        glOptions="translucent",
        parentItem=None,
    ) -> None:
        super().__init__(parentItem=parentItem)
        self.__size = size
        self.__color = np.array(color, dtype=np.float32).clip(0, 1)
        self.__lineWidth = lineWidth
        self.antialias = antialias
        self.setGLOptions(glOptions)
        self.line_vertices = make_grid_data(self.__size, spacing)
        x, y = size
        self.plane_vertices = np.array(
            [
                -x / 2.0,
                -y / 2.0,
                0,
                -x / 2.0,
                y / 2.0,
                0,
                x / 2.0,
                -y / 2.0,
                0,
                x / 2.0,
                y / 2.0,
                0,
            ],
            dtype=np.float32,
        )
        self.rotate(90, 1, 0, 0)
        self.setDepthValue(-1)

    def initializeGL(self):
        self.shader = Shader(vertex_shader, fragment_shader)
        self.vao = VAO()
        self.vbo = VBO(data=[self.line_vertices, self.plane_vertices], size=[3, 3])

    def paint(self, model_matrix=QMatrix4x4()):
        self.setupGLState()

        if self.antialias:
            GL.glEnable(GL.GL_LINE_SMOOTH)
            GL.glHint(GL.GL_LINE_SMOOTH_HINT, GL.GL_NICEST)
        GL.glLineWidth(self.__lineWidth)

        self.shader.set_uniform("view", self.proj_view_matrix().data(), "mat4")
        self.shader.set_uniform("model", model_matrix.data(), "mat4")

        with self.shader:
            self.vao.bind()
            self.shader.set_uniform("objColor1", self.__color, "vec4")
            self.vbo.setAttrPointer(1, attr_id=0)
            GL.glDrawArrays(GL.GL_TRIANGLE_STRIP, 0, 4)

            GL.glDisable(GL.GL_BLEND)
            GL.glDisable(GL.GL_DEPTH_TEST)
            self.shader.set_uniform("objColor1", np.array([0, 0, 0, 1]), "vec4")
            self.vbo.setAttrPointer(0, attr_id=0)
            GL.glDrawArrays(GL.GL_LINES, 0, len(self.line_vertices))
            GL.glEnable(GL.GL_DEPTH_TEST)
