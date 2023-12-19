from PySide6.QtGui import QOpenGLExtraFunctions
from OpenGL import GL


class GLUtils:
    @staticmethod
    def initializeShader(shaderCode, shaderType):
        shaderCode = "#version 330 core\n" + shaderCode
        shaderRef = QOpenGLExtraFunctions().glCreateShader(shaderType)
        QOpenGLExtraFunctions().glShaderSource(shaderRef, shaderCode)
        QOpenGLExtraFunctions().glCompileShader(shaderRef)
        compiled = QOpenGLExtraFunctions().glGetShaderiv(
            shaderRef, GL.GL_COMPILE_STATUS
        )
        if not compiled:
            errorMessage = GL.glGetShaderInfoLog(shaderRef)
            QOpenGLExtraFunctions().glDeleteShader(shaderRef)
            errorMessage = "\n" + errorMessage.decode("utf-8")
            raise Exception(errorMessage)
        return shaderRef

    @staticmethod
    def initializeProgram(vertexShaderCode, fragmentShaderCode):
        vertexShaderRef = GLUtils.initializeShader(
            vertexShaderCode, GL.GL_VERTEX_SHADER
        )
        fragmentSahderRef = GLUtils.initializeShader(
            fragmentShaderCode, GL.GL_FRAGMENT_SHADER
        )

        programRef = QOpenGLExtraFunctions().glCreateProgram()
        QOpenGLExtraFunctions().glAttachShader(programRef, vertexShaderRef)
        QOpenGLExtraFunctions().glAttachShader(programRef, fragmentSahderRef)
        QOpenGLExtraFunctions().glLinkProgram(programRef)
        linked = QOpenGLExtraFunctions().glGetProgramiv(programRef, GL.GL_LINK_STATUS)

        if not linked:
            errorMessage = GL.glGetProgramInfoLog(programRef)
            QOpenGLExtraFunctions().glDeleteProgram(programRef)
            errorMessage = "\n" + errorMessage.decode("utf-8")
            raise Exception(errorMessage)
        return programRef
