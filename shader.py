from OpenGL.GL import shaders
import OpenGL.GL as gl
import numpy as np


class Shader:
    def __init__(
        self, vertex_code, fragment_code, geometry_code=None, uniform_data=None
    ) -> None:
        if geometry_code is not None:
            self.ID = shaders.compileProgram(
                shaders.compileShader(vertex_code, gl.GL_VERTEX_SHADER),
                shaders.compileShader(geometry_code, gl.GL_GEOMETRY_SHADER),
                shaders.compileShader(fragment_code, gl.GL_FRAGMENT_SHADER),
            )
        else:
            self.ID = shaders.compileProgram(
                shaders.compileShader(vertex_code, gl.GL_VERTEX_SHADER),
                shaders.compileShader(fragment_code, gl.GL_FRAGMENT_SHADER),
            )
        if uniform_data is None:
            self.uniform_data = dict()
        else:
            self.uniform_data = uniform_data
        self._in_use = False

    def set_uniform(self, name, data, type: str):
        if self._in_use:
            self.__set_uniform(name, data, type)
        else:
            self.uniform_data[name] = (data, type)

    def use(self):
        gl.glUseProgram(self.ID)
        self._in_use = True
        try:
            for k, d in self.uniform_data.items():
                self.__set_uniform(k, *d)
            self.uniform_data.clear()
        except:
            gl.glUseProgram(0)
            raise

    def unuse(self):
        gl.glUseProgram(0)
        self._in_use = False

    def __set_uniform(self, name, value, type: str, cnt=1):
        if type in ["bool", "int"]:
            gl.glUniform1iv(
                gl.glGetUniformLocation(self.ID, name),
                cnt,
                np.array(value, dtype=np.int32),
            )
        elif type == "sampler2D":
            gl.glUniform1i(
                gl.glGetUniformLocation(self.ID, name),
                np.array(value.unit, dtype=np.int32),
            )
        elif type == "float":
            gl.glUniform1fv(
                gl.glGetUniformLocation(self.ID, name),
                cnt,
                np.array(value, dtype=np.float32),
            )
        elif type == "vec2":
            gl.glUniform2fv(
                gl.glGetUniformLocation(self.ID, name),
                cnt,
                np.array(value, dtype=np.float32),
            )
        elif type == "vec3":
            gl.glUniform3fv(
                gl.glGetUniformLocation(self.ID, name),
                cnt,
                np.array(value, dtype=np.float32),
            )
        elif type == "vec4":
            gl.glUniform4fv(
                gl.glGetUniformLocation(self.ID, name),
                cnt,
                np.array(value, dtype=np.float32),
            )
        elif type == "mat3":
            gl.glUniformMatrix3fv(
                gl.glGetUniformLocation(self.ID, name),
                cnt,
                gl.GL_FALSE,
                np.array(value, dtype=np.float32),
            )
        elif type == "mat4":
            gl.glUniformMatrix4fv(
                gl.glGetUniformLocation(self.ID, name),
                cnt,
                gl.GL_FALSE,
                np.array(value, dtype=np.float32),
            )

    def __del__(self):
        gl.glDeleteProgram(self.ID)

    def __enter__(self):
        self.use()

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is not None:
            print(f"An exception occured: {exc_type}: {exc_val}")
        self.unuse()
