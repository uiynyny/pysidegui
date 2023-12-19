import math
import numpy as np
from OpenGL import GL as gl
from PySide6.QtGui import QMatrix4x4, QVector3D
from bufferobject import EBO, VAO, VBO

from glgraphicsitem import GLGraphicsItem
from shader import Shader


def cone(radius, height, slices=12):
    slices = max(3, slices)
    vertices = np.zeros((slices + 2, 3), dtype="f4")
    vertices[-2] = [0, 0, height]
    step = 360 / slices  # 圆每一段的角度
    for i in range(0, slices):
        p = step * i * 3.14159 / 180  # 转为弧度
        vertices[i] = [radius * math.cos(p), radius * math.sin(p), 0]
    # 构造圆锥的面索引
    indices = np.zeros((slices * 6,), dtype=np.uint32)
    for i in range(0, slices):
        indices[i * 6 + 0] = i
        indices[i * 6 + 1] = (i + 1) % slices
        indices[i * 6 + 2] = slices
        indices[i * 6 + 3] = i
        indices[i * 6 + 5] = (i + 1) % slices
        indices[i * 6 + 4] = slices + 1
    return vertices, indices.reshape(-1, 3)


def direction_matrixs(starts, ends):
    arrows = ends - starts
    arrows = arrows.reshape(-1, 3)
    # 处理零向量，归一化
    arrow_lens = np.linalg.norm(arrows, axis=1)
    zero_idxs = arrow_lens < 1e-3
    arrows[zero_idxs] = [0, 0, 1e-3]
    arrow_lens[zero_idxs] = 1e-3
    arrows = arrows / arrow_lens[:, np.newaxis]
    # 构造标准箭头到目标箭头的旋转矩阵
    e = np.zeros_like(arrows)
    e[arrows[:, 0] == 0, 0] = 1
    e[arrows[:, 0] != 0, 1] = 1
    b1 = np.cross(arrows, e)  # 第一个正交向量 (n, 3)
    b1 = b1 / np.linalg.norm(b1, axis=1, keepdims=True)  # 单位化
    b2 = np.cross(arrows, b1)  # 第二个正交单位向量 (n, 3)
    transforms = np.stack(
        (b1, b2, arrows, ends.reshape(-1, 3)), axis=1,dtype=np.float32
    )  # (n, 4(new), 3)
    # 转化成齐次变换矩阵
    transforms = np.pad(
        transforms, ((0, 0), (0, 0), (0, 1)), mode="constant", constant_values=0
    )  # (n, 4, 4)
    transforms[:, 3, 3] = 1
    # 将 arrow_vert(n, 3) 变换至目标位置
    # vertexes = vertexes @ transforms  #  (n, 3)
    return transforms.copy()


class GLAxisItem(GLGraphicsItem):
    """
    Displays three lines indicating origin and orientation of local coordinate system.
    """

    stPos = np.array(
        [
            # positions
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ],
        dtype="f4",
    )

    endPos = np.array(
        [
            # positions
            1.0,
            0.0,
            0.0,
            0.0,
            1.0,
            0.0,
            0.0,
            0.0,
            1.0,
        ],
        dtype="f4",
    )

    colors = np.array(
        [
            1.0,
            0.0,
            0.0,
            0.0,
            1.0,
            0.0,
            0.0,
            0.0,
            1.0,
        ],
        dtype="f4",
    )

    def __init__(
        self,
        size=np.array([1.0, 1.0, 1.0],dtype=np.float32),
        width=1.5,
        tip_size=1,
        antialias=True,
        glOptions="opaque",
        parentItem: GLGraphicsItem = None,
        depthValue: int = 0,
    ):
        super().__init__(parentItem, depthValue)
        self.__size = size
        self.__width = width
        self.antialias = antialias
        self.setGLOptions(glOptions)
        self.cone_vertices, self.cone_indices = cone(
            0.12 * width * tip_size, 0.3 * width * tip_size
        )

    def setSize(self, size):
        self.__size = size
        self.update()

    def size(self):
        return self.__size

    def initializeGL(self):
        self.shader = Shader(vertex_shader, fragment_shader, geometry_shader)
        self.shader_cone = Shader(vertex_shader_cone, fragment_shader)
        self.vao_line = VAO()
        self.vbo1 = VBO(
            data=[self.stPos, self.endPos, self.colors],
            size=[3, 3, 3],
        )
        self.vbo1.setAttrPointer([0, 1, 2])

        # cone
        self.transforms = direction_matrixs(
            self.stPos.reshape(-1, 3) * self.__size,
            self.endPos.reshape(-1, 3) * self.__size,
        )
        self.vao_cone = VAO()

        self.vbo2 = VBO(
            [self.cone_vertices, self.transforms],
            [3, [4, 4, 4, 4]],
        )
        self.vbo2.setAttrPointer([0, 1], divisor=[0, 1], attr_id=[0, [1, 2, 3, 4]])

        self.vbo1.bind()
        self.vbo1.setAttrPointer(2, divisor=1, attr_id=5)

        self.ebo = EBO(self.cone_indices)

    def paint(self, model_matrix=QMatrix4x4()):
        self.setupGLState()

        if self.antialias:
            gl.glEnable(gl.GL_LINE_SMOOTH)
            gl.glHint(gl.GL_LINE_SMOOTH_HINT, gl.GL_NICEST)
        gl.glLineWidth(self.__width)

        with self.shader:
            self.shader.set_uniform("sizev3", self.size(), "vec3")
            self.shader.set_uniform("view", self.proj_view_matrix().data(), "mat4")
            self.shader.set_uniform("model", model_matrix.data(), "mat4")
            self.vao_line.bind()
            gl.glDrawArrays(gl.GL_POINTS, 0, 3)

        gl.glEnable(gl.GL_CULL_FACE)
        gl.glCullFace(gl.GL_BACK)
        with self.shader_cone:
            self.shader_cone.set_uniform("view", self.proj_view_matrix().data(), "mat4")
            self.shader_cone.set_uniform("model", model_matrix.data(), "mat4")
            self.vao_cone.bind()
            gl.glDrawElementsInstanced(gl.GL_TRIANGLES, self.cone_indices.size, gl.GL_UNSIGNED_INT, None, 3)
        gl.glDisable(gl.GL_CULL_FACE)

vertex_shader = """
#version 330 core

uniform mat4 model;
uniform mat4 view;
uniform vec3 sizev3;

layout (location = 0) in vec3 stPos;
layout (location = 1) in vec3 endPos;
layout (location = 2) in vec3 iColor;

out V_OUT {
    vec4 endPos;
    vec3 color;
} v_out;

void main() {
    mat4 matrix = view * model;
    gl_Position =  matrix * vec4(stPos * sizev3, 1.0);
    v_out.endPos = matrix * vec4(endPos * sizev3, 1.0);
    v_out.color = iColor;
}
"""

vertex_shader_cone = """
#version 330 core

uniform mat4 model;
uniform mat4 view;

layout (location = 0) in vec3 iPos;
layout (location = 1) in vec4 row1;
layout (location = 2) in vec4 row2;
layout (location = 3) in vec4 row3;
layout (location = 4) in vec4 row4;
layout (location = 5) in vec3 iColor;
out vec3 oColor;

void main() {
    mat4 transform = mat4(row1, row2, row3, row4);
    gl_Position =  view * model * transform * vec4(iPos, 1.0);
    oColor = iColor;
}
"""

geometry_shader = """
#version 330 core
layout(points) in;
layout(line_strip, max_vertices = 2) out;

in V_OUT {
    vec4 endPos;
    vec3 color;
} gs_in[];
out vec3 oColor;

void main() {
    oColor = gs_in[0].color;
    gl_Position = gl_in[0].gl_Position;
    EmitVertex();
    gl_Position = gs_in[0].endPos;
    EmitVertex();
    EndPrimitive();
}
"""

fragment_shader = """
#version 330 core

in vec3 oColor;
out vec4 fragColor;

void main() {
    fragColor = vec4(oColor, 1.0f);
}
"""
