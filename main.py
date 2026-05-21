import sys
import os
import subprocess
import io
import time
import json
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QPushButton, QLabel, QFileDialog, 
                             QSlider, QComboBox, QCheckBox, QGraphicsView, 
                             QGraphicsScene, QGraphicsPixmapItem, QMessageBox,
                             QDialog, QFormLayout, QStackedWidget, QSpinBox,
                             QListWidget, QListWidgetItem, QGridLayout, QScrollArea)
from PyQt6.QtOpenGLWidgets import QOpenGLWidget
from PyQt6.QtCore import Qt, QTimer, pyqtSignal, QSize, QRunnable, QThreadPool, QObject
from PyQt6.QtGui import QPixmap, QImage, QPainter, QColor, QPainterPath, QIcon, QTransform
import rawpy
import numpy as np
from PIL import Image, ImageCms
from OpenGL.GL import *
from OpenGL.GL import shaders
import exifread
import math

def get_x11_icc_profile():
    try:
        output = subprocess.check_output(['xprop', '-root', '-notype', '32c', '_ICC_PROFILE'], stderr=subprocess.DEVNULL)
        output = output.decode('utf-8')
        if '_ICC_PROFILE' not in output or 'not found' in output:
            return None
        
        hex_str = output.split('=')[1].strip()
        byte_list = [int(x.strip(), 16) for x in hex_str.split(',')]
        return bytes(byte_list)
    except Exception as e:
        return None

def get_colord_icc_file():
    try:
        output = subprocess.check_output(['colormgr', 'get-devices-by-kind', 'display'], stderr=subprocess.DEVNULL).decode('utf-8')
        device_path = None
        for line in output.split('\n'):
            if line.strip().startswith('Object Path:'):
                device_path = line.split(':', 1)[1].strip()
                break
        
        if device_path:
            profile_output = subprocess.check_output(['colormgr', 'device-get-default-profile', device_path], stderr=subprocess.DEVNULL).decode('utf-8')
            for p_line in profile_output.split('\n'):
                if p_line.strip().startswith('Filename:'):
                    return p_line.split(':', 1)[1].strip()
    except Exception as e:
        return None
    return None

def pil2pixmap(im):
    im = im.convert("RGBA")
    data = im.tobytes("raw", "BGRA")
    qim = QImage(data, im.width, im.height, QImage.Format.Format_ARGB32)
    # Important: keep a reference to data to prevent garbage collection before QImage is used
    qim.nd = data
    pixmap = QPixmap.fromImage(qim)
    return pixmap

class ImageGraphicsView(QGraphicsView):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)

    def wheelEvent(self, event):
        zoom_in_factor = 1.15
        zoom_out_factor = 1 / zoom_in_factor

        if event.angleDelta().y() > 0:
            zoom_factor = zoom_in_factor
        else:
            zoom_factor = zoom_out_factor

        self.scale(zoom_factor, zoom_factor)

class GLImageView(QOpenGLWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.image_data = None
        self.img_width = 0
        self.img_height = 0
        
        self.exposure = 1.0
        self.gamma = 2.22
        
        self.zoom = 1.0
        self.pan_x = 0.0
        self.pan_y = 0.0
        self.last_pos = None

    def initializeGL(self):
        is_gles = self.context().isOpenGLES()
        version_str = "#version 300 es\n" if is_gles else "#version 330 core\n"
        frag_prec = "precision highp float;\nprecision highp sampler3D;\n" if is_gles else ""
        
        vert_src = version_str + """
            layout (location = 0) in vec2 aPos;
            layout (location = 1) in vec2 aTexCoord;
            out vec2 TexCoord;
            uniform mat4 transform;
            void main() {
                gl_Position = transform * vec4(aPos, 0.0, 1.0);
                TexCoord = aTexCoord;
            }
        """
        
        frag_src = version_str + frag_prec + """
            out vec4 FragColor;
            in vec2 TexCoord;
            uniform sampler2D ourTexture;
            uniform sampler2D toneCurveTex;
            uniform float exposure;
            uniform float gamma;
            uniform vec3 wb;
            uniform float saturation;
            uniform float hl_protect;
            uniform sampler3D colorLutTex;
            uniform float encodePower;
            
            void main() {
                vec4 texColor = texture(ourTexture, TexCoord);
                vec3 color = texColor.rgb * wb * exposure;
                
                // Tone Curve
                color.r = texture(toneCurveTex, vec2(clamp(color.r, 0.0, 1.0), 0.5)).r;
                color.g = texture(toneCurveTex, vec2(clamp(color.g, 0.0, 1.0), 0.5)).r;
                color.b = texture(toneCurveTex, vec2(clamp(color.b, 0.0, 1.0), 0.5)).r;
                
                float luma = dot(color, vec3(0.299, 0.587, 0.114));
                
                if (hl_protect > 0.0) {
                    float t = clamp((luma - 0.5) / 0.5, 0.0, 1.0);
                    float new_luma = luma - (t * t * 0.2 * hl_protect);
                    color = color * (new_luma / max(luma, 1e-6));
                    luma = new_luma;
                }
                
                // Saturation
                color = mix(vec3(luma), color, saturation);
                
                color = clamp(color, 0.0, 1.0);
                
                // Encode for 3D LUT lookup
                color = pow(color, vec3(encodePower));
                
                // 3D LUT transform from Source Profile to Display Profile
                color = texture(colorLutTex, color).rgb;
                
                FragColor = vec4(color, 1.0);
            }
        """
        
        self.shader_program = shaders.compileProgram(
            shaders.compileShader(vert_src, GL_VERTEX_SHADER),
            shaders.compileShader(frag_src, GL_FRAGMENT_SHADER)
        )
        
        vertices = np.array([
             1.0,  1.0,   1.0, 0.0,
             1.0, -1.0,   1.0, 1.0,
            -1.0, -1.0,   0.0, 1.0,
            -1.0,  1.0,   0.0, 0.0 
        ], dtype=np.float32)
        
        indices = np.array([0, 1, 3, 1, 2, 3], dtype=np.uint32)
        
        self.VAO = glGenVertexArrays(1)
        self.VBO = glGenBuffers(1)
        self.EBO = glGenBuffers(1)
        
        glBindVertexArray(self.VAO)
        glBindBuffer(GL_ARRAY_BUFFER, self.VBO)
        glBufferData(GL_ARRAY_BUFFER, vertices.nbytes, vertices, GL_STATIC_DRAW)
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, self.EBO)
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices.nbytes, indices, GL_STATIC_DRAW)
        
        glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 4 * 4, ctypes.c_void_p(0))
        glEnableVertexAttribArray(0)
        glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 4 * 4, ctypes.c_void_p(2 * 4))
        glEnableVertexAttribArray(1)
        
        self.texture = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, self.texture)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        
        self.curve_texture = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, self.curve_texture)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        
        self.color_lut_texture = glGenTextures(1)
        glBindTexture(GL_TEXTURE_3D, self.color_lut_texture)
        glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
        glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)
        glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE)
        glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        # Initialize default identity LUT
        ident_lut = np.zeros((2, 2, 2, 3), dtype=np.uint8)
        self.encode_power = 1.0
        self.set_color_lut(ident_lut, 1.0)
        
        if hasattr(self, 'pending_curve'):
            self.set_curve(self.pending_curve)
        else:
            self.set_curve([(0.0, 0.0), (1.0, 1.0)])
        
    def set_curve(self, points):
        if not hasattr(self, 'curve_texture'):
            self.pending_curve = points
            return
        self.makeCurrent()
        dense_y = interpolate_curve(points, 256)
        glBindTexture(GL_TEXTURE_2D, self.curve_texture)
        glPixelStorei(GL_UNPACK_ALIGNMENT, 1)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_R32F, 256, 1, 0, GL_RED, GL_FLOAT, dense_y)
        self.update()

    def set_color_lut(self, lut_data_uint8, encode_power):
        self.makeCurrent()
        self.encode_power = encode_power
        glBindTexture(GL_TEXTURE_3D, self.color_lut_texture)
        glPixelStorei(GL_UNPACK_ALIGNMENT, 1)
        lut_size = lut_data_uint8.shape[0]
        glTexImage3D(GL_TEXTURE_3D, 0, GL_RGB8, lut_size, lut_size, lut_size, 0, GL_RGB, GL_UNSIGNED_BYTE, lut_data_uint8)
        self.update()

    def set_image(self, data, w, h):
        self.image_data = data
        self.img_width = w
        self.img_height = h
        self.makeCurrent()
        glBindTexture(GL_TEXTURE_2D, self.texture)
        glPixelStorei(GL_UNPACK_ALIGNMENT, 1)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB16, w, h, 0, GL_RGB, GL_UNSIGNED_SHORT, data)
        self.update()
        
    def paintGL(self):
        glClearColor(0.2, 0.2, 0.2, 1.0)
        glClear(GL_COLOR_BUFFER_BIT)
        if self.image_data is None: return
        glUseProgram(self.shader_program)
        glUniform1f(glGetUniformLocation(self.shader_program, "exposure"), self.exposure)
        glUniform1f(glGetUniformLocation(self.shader_program, "gamma"), self.gamma)
        
        glUniform1f(glGetUniformLocation(self.shader_program, "saturation"), 1.0 + getattr(self, 'saturation', 0.0))
        glUniform1f(glGetUniformLocation(self.shader_program, "hl_protect"), getattr(self, 'hl_protect', 0.0))
        
        glUniform1f(glGetUniformLocation(self.shader_program, "encodePower"), getattr(self, 'encode_power', 1.0))
        
        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D, self.texture)
        glUniform1i(glGetUniformLocation(self.shader_program, "ourTexture"), 0)

        glActiveTexture(GL_TEXTURE1)
        glBindTexture(GL_TEXTURE_2D, self.curve_texture)
        glUniform1i(glGetUniformLocation(self.shader_program, "toneCurveTex"), 1)
        
        glActiveTexture(GL_TEXTURE2)
        glBindTexture(GL_TEXTURE_3D, getattr(self, 'color_lut_texture', 0))
        glUniform1i(glGetUniformLocation(self.shader_program, "colorLutTex"), 2)
        
        glActiveTexture(GL_TEXTURE0)
        
        wb_loc = glGetUniformLocation(self.shader_program, "wb")
        if hasattr(self, 'wb_mults'):
            glUniform3f(wb_loc, self.wb_mults[0], self.wb_mults[1], self.wb_mults[2])
        else:
            glUniform3f(wb_loc, 1.0, 1.0, 1.0)
        
        widget_ar = self.width() / self.height() if self.height() > 0 else 1.0
        img_ar = self.img_width / self.img_height if self.img_height > 0 else 1.0
        
        scale_x = self.zoom
        scale_y = self.zoom
        if widget_ar > img_ar: scale_x *= img_ar / widget_ar
        else: scale_y *= widget_ar / img_ar
            
        transform = np.array([
            [scale_x, 0.0, 0.0, 0.0],
            [0.0, scale_y, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [self.pan_x, -self.pan_y, 0.0, 1.0]
        ], dtype=np.float32)
        glUniformMatrix4fv(glGetUniformLocation(self.shader_program, "transform"), 1, GL_FALSE, transform)
        
        glBindVertexArray(self.VAO)
        glBindTexture(GL_TEXTURE_2D, self.texture)
        glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, None)
        
    def wheelEvent(self, event):
        old_zoom = self.zoom
        zoom_factor = 1.15
        if event.angleDelta().y() > 0: self.zoom *= zoom_factor
        else: self.zoom /= zoom_factor
        
        # Scale pan to keep the center of the screen fixed
        ratio = self.zoom / old_zoom
        self.pan_x *= ratio
        self.pan_y *= ratio
        
        self.update()
        
    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton: self.last_pos = event.pos()
            
    def mouseMoveEvent(self, event):
        if event.buttons() & Qt.MouseButton.LeftButton and self.last_pos:
            dx = event.pos().x() - self.last_pos.x()
            dy = event.pos().y() - self.last_pos.y()
            self.pan_x += (dx / self.width()) * 2.0
            self.pan_y += (dy / self.height()) * 2.0
            self.last_pos = event.pos()
            self.update()
            
    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton: self.last_pos = None

    def fit_in_view(self):
        self.zoom = 1.0
        self.pan_x = 0.0
        self.pan_y = 0.0
        self.update()

class SettingsDialog(QDialog):
    def __init__(self, parent):
        super().__init__(parent)
        self.setWindowTitle("Settings")
        self.resize(450, 200)
        self.parent_editor = parent
        
        layout = QFormLayout(self)
        
        self.cmb_mode = QComboBox()
        self.cmb_mode.addItems([
            "Full rawpy processing (Slow)", 
            "Linear Cache Math (Medium)", 
            "LUT Optimization (Fast)",
            "OpenGL GPU Shader (Ultra Fast)"
        ])
        self.cmb_mode.setCurrentIndex(parent.processing_mode)
        layout.addRow("Processing Mode:", self.cmb_mode)
        
        self.chk_fast_preview = QCheckBox("Fast Preview (Half Size)")
        self.chk_fast_preview.setChecked(parent.fast_preview)
        layout.addRow("", self.chk_fast_preview)
        
        self.btn_profile = QPushButton("Set Custom Display Profile")
        self.btn_profile.clicked.connect(self.set_custom_profile)
        
        self.lbl_custom_prof = QLabel(os.path.basename(parent.custom_profile_path) if parent.custom_profile_path else "None")
        
        prof_layout = QHBoxLayout()
        prof_layout.addWidget(self.btn_profile)
        prof_layout.addWidget(self.lbl_custom_prof)
        layout.addRow("Custom Display Profile:", prof_layout)
        
        self.btn_ok = QPushButton("OK")
        self.btn_ok.clicked.connect(self.accept)
        layout.addRow("", self.btn_ok)
        
    def set_custom_profile(self):
        file_name, _ = QFileDialog.getOpenFileName(self, "Select ICC Profile", "", "ICC Profiles (*.icc *.icm);;All Files (*)")
        if file_name:
            self.parent_editor.custom_profile_path = file_name
            self.lbl_custom_prof.setText(os.path.basename(file_name))

class ClickableLabel(QLabel):
    clicked = pyqtSignal()
    def mousePressEvent(self, event):
        self.clicked.emit()
        super().mousePressEvent(event)

class HistogramWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(250, 150)
        self.hist_r = np.zeros(256, dtype=np.uint64)
        self.hist_g = np.zeros(256, dtype=np.uint64)
        self.hist_b = np.zeros(256, dtype=np.uint64)
        self.hist_y = np.zeros(256, dtype=np.uint64)

    def update_hist(self, r, g, b):
        self.hist_r = r
        self.hist_g = g
        self.hist_b = b
        # Luminance approximation: Y = 0.299R + 0.587G + 0.114B
        self.hist_y = (0.299 * r + 0.587 * g + 0.114 * b).astype(np.uint64)
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        
        rect = self.rect()
        w = rect.width()
        h = rect.height()
        
        # Draw background
        painter.fillRect(rect, Qt.GlobalColor.black)
        
        # Partition lines
        # Blacks (0-10%), Shadows (10-40%), Midtones (40-70%), Highlights (70-90%), Whites (90-100%)
        partition_boundaries = [0.1, 0.4, 0.7, 0.9]
        
        painter.setPen(QColor(80, 80, 80))
        for p in partition_boundaries:
            px = int(p * w)
            painter.drawLine(px, 0, px, h)

        # Global normalization: Ignore clipping peaks at 0 and 255 to keep midtone detail visible
        def get_safe_max(data):
            if data.size < 256: return 1
            # Look only at the "meat" of the histogram (bins 1-254)
            return data[1:255].max()
            
        global_max = max(get_safe_max(self.hist_r), 
                         get_safe_max(self.hist_g), 
                         get_safe_max(self.hist_b), 
                         get_safe_max(self.hist_y), 1)

        def draw_channel(data, color, alpha=150):
            if global_max == 0: return
            path = QPainterPath()
            path.moveTo(0, h)
            
            for i in range(256):
                x = (i / 255.0) * w
                # Scale relative to the safe max, but cap at the widget height
                val = min((data[i] / global_max) * h, h)
                path.lineTo(x, h - val)
            
            path.lineTo(w, h)
            painter.setBrush(QColor(color.red(), color.green(), color.blue(), alpha))
            painter.setPen(Qt.PenStyle.NoPen)
            painter.drawPath(path)

        draw_channel(self.hist_r, QColor(255, 0, 0))
        draw_channel(self.hist_g, QColor(0, 255, 0))
        draw_channel(self.hist_b, QColor(0, 0, 255))
        draw_channel(self.hist_y, QColor(255, 255, 255), alpha=100)
        
        # Draw outline for Y
        painter.setPen(QColor(255, 255, 255, 200))
        painter.setBrush(Qt.BrushStyle.NoBrush)
        if global_max > 0:
            for i in range(255):
                x1 = (i / 255.0) * w
                y1 = h - min((self.hist_y[i] / global_max) * h, h)
                x2 = ((i+1) / 255.0) * w
                y2 = h - min((self.hist_y[i+1] / global_max) * h, h)
                painter.drawLine(int(x1), int(y1), int(x2), int(y2))

from PyQt6.QtGui import QPen

def interpolate_curve(points, n_samples=256):
    x = [p[0] for p in points]
    y = [p[1] for p in points]
    dense_x = np.linspace(0, 1, n_samples)
    try:
        from scipy.interpolate import pchip_interpolate
        dense_y = pchip_interpolate(x, y, dense_x)
    except ImportError:
        dense_y = np.interp(dense_x, x, y)
    return np.clip(dense_y, 0.0, 1.0).astype(np.float32)

def apply_saturation_and_hl(arr, sat_mult, hl_protect):
    was_uint8 = False
    if arr.dtype == np.uint8:
        arr = arr.astype(np.float32) / 255.0
        was_uint8 = True
        
    luma = 0.299 * arr[:,:,0] + 0.587 * arr[:,:,1] + 0.114 * arr[:,:,2]
    res = arr.copy()
    
    if hl_protect > 0.0:
        t = np.clip((luma - 0.5) / 0.5, 0.0, 1.0)
        new_luma = luma - (t * t * 0.2 * hl_protect)
        mult = new_luma / np.maximum(luma, 1e-6)
        res[:,:,0] *= mult
        res[:,:,1] *= mult
        res[:,:,2] *= mult
        luma = new_luma
        
    if sat_mult != 1.0:
        res[:,:,0] = luma + (res[:,:,0] - luma) * sat_mult
        res[:,:,1] = luma + (res[:,:,1] - luma) * sat_mult
        res[:,:,2] = luma + (res[:,:,2] - luma) * sat_mult
        
    if was_uint8:
        return np.clip(res * 255.0, 0, 255).astype(np.uint8)
    return np.clip(res, 0.0, 1.0)

class ToneCurveWidget(QWidget):
    changed = pyqtSignal()
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(250, 200)
        self.points = [(0.0, 0.0), (1.0, 1.0)]
        self.active_idx = -1
        self.drag_mode = False

    def get_points(self):
        return self.points

    def set_points(self, points):
        self.points = sorted(points, key=lambda p: p[0])
        self.update()
        self.changed.emit()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        w, h = self.width(), self.height()
        
        painter.fillRect(self.rect(), QColor(40, 40, 40))
        painter.setPen(QColor(80, 80, 80))
        for i in range(1, 4):
            x = int(i * w / 4)
            y = int(i * h / 4)
            painter.drawLine(x, 0, x, h)
            painter.drawLine(0, y, w, y)
            
        if len(self.points) > 1:
            path = QPainterPath()
            path.moveTo(self.points[0][0] * w, h - self.points[0][1] * h)
            dense_x = np.linspace(0, 1, 200)
            dense_y = interpolate_curve(self.points, 200)
            for dx, dy in zip(dense_x, dense_y):
                path.lineTo(dx * w, h - dy * h)
            painter.setPen(QPen(QColor(200, 200, 200), 2))
            painter.drawPath(path)
            
        painter.setBrush(QColor(150, 150, 150))
        for i, (px, py) in enumerate(self.points):
            if i == self.active_idx:
                painter.setPen(QPen(QColor(255, 255, 255), 2))
            else:
                painter.setPen(Qt.PenStyle.NoPen)
            cx, cy = px * w, h - py * h
            painter.drawEllipse(int(cx - 4), int(cy - 4), 8, 8)

    def mousePressEvent(self, event):
        pos = event.pos()
        w, h = self.width(), self.height()
        click_x, click_y = pos.x() / w, 1.0 - pos.y() / h
        
        best_dist = 0.05
        best_idx = -1
        for i, (px, py) in enumerate(self.points):
            dist = math.hypot(px - click_x, py - click_y)
            if dist < best_dist:
                best_dist = dist
                best_idx = i
                
        if best_idx != -1:
            self.active_idx = best_idx
            if event.button() == Qt.MouseButton.RightButton and self.active_idx not in (0, len(self.points)-1):
                self.points.pop(self.active_idx)
                self.active_idx = -1
                self.changed.emit()
            else:
                self.drag_mode = True
        elif event.button() == Qt.MouseButton.LeftButton:
            self.points.append((click_x, click_y))
            self.points.sort(key=lambda p: p[0])
            self.active_idx = self.points.index((click_x, click_y))
            self.drag_mode = True
            self.changed.emit()
        self.update()

    def mouseMoveEvent(self, event):
        if self.drag_mode and self.active_idx != -1:
            pos = event.pos()
            w, h = self.width(), self.height()
            x, y = pos.x() / w, 1.0 - pos.y() / h
            x = max(0.0, min(1.0, x))
            y = max(0.0, min(1.0, y))
            
            if self.active_idx > 0:
                x = max(x, self.points[self.active_idx-1][0] + 0.01)
            if self.active_idx < len(self.points) - 1:
                x = min(x, self.points[self.active_idx+1][0] - 0.01)
                
            if self.active_idx == 0: x = 0.0
            elif self.active_idx == len(self.points) - 1: x = 1.0
                
            self.points[self.active_idx] = (x, y)
            self.update()
            self.changed.emit()

    def mouseReleaseEvent(self, event):
        self.drag_mode = False

def process_thumbnail_task(file_path, index):
    import rawpy
    from PIL import Image
    import io
    try:
        with rawpy.imread(file_path) as raw:
            try:
                thumb = raw.extract_thumb()
                if thumb.format == rawpy.ThumbFormat.JPEG:
                    img = Image.open(io.BytesIO(thumb.data))
                    flip = raw.sizes.flip
                    if flip == 3:
                        img = img.transpose(Image.ROTATE_180)
                    elif flip == 5: # 90 CCW in EXIF (-90 CW)
                        img = img.transpose(Image.ROTATE_90)
                    elif flip == 6: # 90 CW in EXIF
                        img = img.transpose(Image.ROTATE_270)
                    
                    img.thumbnail((160, 160), Image.Resampling.LANCZOS)
                    
                    out = io.BytesIO()
                    img.save(out, format='JPEG')
                    return (index, out.getvalue())
            except rawpy.LibRawNoThumbnailError:
                pass
    except Exception:
        pass
    return (index, None)

from concurrent.futures import ProcessPoolExecutor, as_completed, wait, FIRST_COMPLETED
import threading
try:
    import psutil as _psutil
except ImportError:
    _psutil = None

# ---------------------------------------------------------------------------
# Top-level helpers for multiprocessing (must be picklable, i.e. not methods)
# ---------------------------------------------------------------------------
def _kelvin_to_rgb_pure(kelvin):
    """Pure-function version of kelvin_to_rgb for use in worker processes."""
    import math, numpy as np
    temp = kelvin / 100.0
    if temp <= 66:
        r = 255
        g = 99.4708025861 * math.log(temp) - 161.1195681661
        b = 0 if temp <= 19 else 138.5177312231 * math.log(temp - 10) - 305.0447927307
    else:
        r = 329.698727446 * math.pow(temp - 60, -0.1332047592)
        g = 288.1221695283 * math.pow(temp - 60, -0.0755148492)
        b = 255
    return float(np.clip(r, 0, 255)), float(np.clip(g, 0, 255)), float(np.clip(b, 0, 255))


def _batch_export_task(args):
    """
    Top-level (picklable) export function executed in a worker process.
    args = (file_path, settings, out_path, pil_fmt, system_icc_paths)
    Returns (basename, error_str_or_None)
    """
    import rawpy as _rawpy
    import numpy as np
    from PIL import Image, ImageCms
    import os

    file_path, settings, out_path, pil_fmt, system_icc_paths = args
    basename = os.path.basename(file_path)
    try:
        ev                 = settings['exposure'] / 6.0
        exposure_mult      = 2.0 ** ev
        gamma_val          = settings['gamma'] / 100.0
        use_camera_wb      = settings['wb'] == 'Camera'
        use_auto_wb        = settings['wb'] == 'Auto'
        cs_text            = settings['colorspace']

        if cs_text == 'sRGB':
            out_cs = _rawpy.ColorSpace.sRGB
            base_gamma = 2.22
        elif cs_text == 'Adobe RGB':
            out_cs = _rawpy.ColorSpace.Adobe
            base_gamma = 2.2
        else:
            out_cs = _rawpy.ColorSpace.ProPhoto
            base_gamma = 1.8

        with _rawpy.imread(file_path) as raw:
            export_lin = raw.postprocess(
                use_camera_wb=use_camera_wb,
                use_auto_wb=use_auto_wb,
                half_size=False,
                exp_shift=1.0,
                gamma=(1, 1),
                output_color=out_cs,
                output_bps=16,
                no_auto_bright=True,
            )

        # White balance
        dt, dtint = settings['temp'], settings['tint']
        r_ref, g_ref, b_ref = _kelvin_to_rgb_pure(6500)
        r_off, g_off, b_off = _kelvin_to_rgb_pure(float(np.clip(6500 + dt, 2000, 12000)))
        g_off *= (1.0 - dtint / 200.0)
        wb = [r_ref / r_off, g_ref / g_off, b_ref / b_off]

        arr = export_lin.astype(np.float32) / 65535.0
        for ch in range(3):
            arr[:, :, ch] *= wb[ch] * exposure_mult

        # Tonal adjustments
        curve_points = settings.get('tone_curve', [[0.0, 0.0], [1.0, 1.0]])
        lut = interpolate_curve(curve_points, 65536)
        x_idx = np.clip(arr * 65535, 0, 65535).astype(np.int32)
        arr = lut[x_idx]

        # Saturation and HL
        sat_mult = 1.0 + settings['saturation'] / 100.0
        hl_protect = settings.get('hl_protect', 0) / 100.0
        if sat_mult != 1.0 or hl_protect > 0.0:
            arr = apply_saturation_and_hl(arr, sat_mult, hl_protect)

        # Gamma + quantise
        creative_gamma_adjustment = 2.22 / gamma_val
        encode_power = creative_gamma_adjustment / base_gamma
        arr = np.power(np.clip(arr, 1e-6, 1.0), encode_power) * 255.0
        img = Image.fromarray(arr.astype(np.uint8))

        # ICC profile
        icc_bytes = None
        icc_path  = system_icc_paths.get(cs_text)
        if icc_path and os.path.exists(icc_path):
            with open(icc_path, 'rb') as f:
                icc_bytes = f.read()
        elif cs_text == 'sRGB':
            icc_bytes = ImageCms.createProfile('sRGB').tobytes()

        save_kw = {'icc_profile': icc_bytes} if icc_bytes else {}
        if pil_fmt == 'JPEG':
            save_kw['quality']     = 95
            save_kw['subsampling'] = 0
        img.save(out_path, format=pil_fmt, **save_kw)
        return (basename, None)
    except Exception as exc:
        return (basename, str(exc))


class BatchExportSignals(QObject):
    progress = pyqtSignal(int, int, int)   # (done, total, active_workers)
    finished = pyqtSignal(list, list, int) # (skipped_list, failed_list, done_count)


class BatchExportWorker(QRunnable):
    def __init__(self, tasks, skipped, stop_event):
        """
        tasks      – list of arg-tuples for _batch_export_task
        skipped    – list of basenames that had no saved settings
        stop_event – threading.Event; set it to request cancellation
        """
        super().__init__()
        self.tasks      = tasks
        self.skipped    = skipped
        self.stop_event = stop_event
        self.signals    = BatchExportSignals()

    def run(self):
        import time
        import collections

        total      = len(self.tasks)
        failed     = []
        done       = 0
        task_queue = list(self.tasks)   # tasks not yet submitted
        active     = {}                 # future -> task
        max_cap    = os.cpu_count()     # hard upper bound (all logical CPUs)
        limit      = max(1, max_cap // 4)  # start at 25% of logical CPUs

        MEM_PERCENT_LIMIT   = 75.0
        MEM_WINDOW_SECS     = 15.0   # rolling window to observe peak memory
        MEM_SAMPLE_INTERVAL = 0.25  # background sampler interval (seconds)
        SCALE_DOWN_COOLDOWN = 2.0   # min seconds between consecutive scale-downs
        SCALE_UP_COOLDOWN   = 5.0   # min seconds between consecutive scale-ups

        # Deque of (timestamp, mem_percent) filled by the sampler thread
        window_size = int(MEM_WINDOW_SECS / MEM_SAMPLE_INTERVAL) + 1
        mem_window  = collections.deque(maxlen=window_size)
        window_lock = threading.Lock()

        # ── Background thread: sample memory every MEM_SAMPLE_INTERVAL seconds ──
        _sampler_stop = threading.Event()

        def _sampler():
            while not _sampler_stop.is_set():
                pct = _psutil.virtual_memory().percent if _psutil else 0.0
                with window_lock:
                    mem_window.append((time.monotonic(), pct))
                time.sleep(MEM_SAMPLE_INTERVAL)

        sampler_thread = threading.Thread(target=_sampler, daemon=True)
        sampler_thread.start()

        def _peak_mem():
            """Return the worst (peak) memory % seen in the rolling window."""
            if _psutil is None:
                return 0.0
            cutoff = time.monotonic() - MEM_WINDOW_SECS
            with window_lock:
                recent = [pct for ts, pct in mem_window if ts >= cutoff]
            return max(recent) if recent else _psutil.virtual_memory().percent

        # Initialise cooldown clocks so first adjustment can happen immediately
        last_scale_down = time.monotonic() - SCALE_DOWN_COOLDOWN
        last_scale_up   = time.monotonic() - SCALE_UP_COOLDOWN

        try:
            with ProcessPoolExecutor(max_workers=max_cap) as executor:
                while (task_queue or active) and not self.stop_event.is_set():
                    peak = _peak_mem()

                    # --- Submit tasks up to the current concurrency limit ---
                    while (task_queue and len(active) < limit
                           and peak < MEM_PERCENT_LIMIT
                           and not self.stop_event.is_set()):
                        task   = task_queue.pop(0)
                        future = executor.submit(_batch_export_task, task)
                        active[future] = task

                    if not active:
                        # Nothing running yet; memory under pressure – wait briefly
                        time.sleep(MEM_SAMPLE_INTERVAL)
                        continue

                    # --- Wait for at least one result (short timeout keeps loop alive) ---
                    finished_set, _ = wait(active.keys(), timeout=0.5,
                                           return_when=FIRST_COMPLETED)

                    for future in finished_set:
                        try:
                            basename, err = future.result()
                            if err:
                                failed.append(f"{basename}: {err}")
                        except Exception as exc:
                            failed.append(str(exc))
                        del active[future]
                        done += 1
                        self.signals.progress.emit(done, total, len(active))

                    # --- Adjust concurrency ceiling using windowed peak + cooldowns ---
                    now  = time.monotonic()
                    peak = _peak_mem()   # refresh after waiting

                    if peak >= MEM_PERCENT_LIMIT:
                        if now - last_scale_down >= SCALE_DOWN_COOLDOWN:
                            limit           = max(1, limit - 1)
                            last_scale_down = now
                    elif limit < max_cap:
                        if now - last_scale_up >= SCALE_UP_COOLDOWN:
                            limit         = min(max_cap, limit + 1)
                            last_scale_up = now

                # If cancelled, cancel any still-pending futures
                if self.stop_event.is_set():
                    for f in list(active.keys()):
                        f.cancel()
        finally:
            _sampler_stop.set()
            sampler_thread.join(timeout=1.0)

        self.signals.finished.emit(self.skipped, failed, done)


class ThumbnailManagerSignals(QObject):
    result = pyqtSignal(int, bytes)

class ThumbnailManager(QRunnable):
    def __init__(self, file_paths):
        super().__init__()
        self.file_paths = file_paths
        self.signals = ThumbnailManagerSignals()

    def run(self):
        # We use a ProcessPoolExecutor to completely bypass the Python GIL
        with ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
            futures = {executor.submit(process_thumbnail_task, path, i): i for i, path in enumerate(self.file_paths)}
            for future in as_completed(futures):
                try:
                    idx, data = future.result()
                    if data:
                        self.signals.result.emit(idx, data)
                except Exception:
                    pass

class RawEditor(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("RAW Image Editor")
        self.resize(1200, 800)
        
        self.raw_path = None
        self.raw_image = None
        self.processed_rgb = None
        self.target_icc_path = None
        
        self.processing_mode = 3 # Default to GPU Shader Optimization
        self.fast_preview = False
        self.linear_cache = None
        self.cache_dirty = True
        self.is_first_load = False
        
        # Per-image settings memory: maps file_path -> dict of slider/combo values
        self.image_settings = {}
        self.current_dir = None
        self.SETTINGS_FILENAME = '.raweditor_settings.json'
        
        self.counts_r = None
        self.counts_g = None
        self.counts_b = None
        
        # Color Management
        self.monitor_profile_bytes = get_x11_icc_profile()
        self.monitor_profile_path = get_colord_icc_file()
        self.custom_profile_path = None
        
        # Threading for thumbnails
        self.thumbnail_pool = QThreadPool()
        # Reserve at least one core for the UI thread to prevent blocking
        self.thumbnail_pool.setMaxThreadCount(max(1, QThreadPool.globalInstance().maxThreadCount() - 1))
        
        # Output color spaces map to system ICC files
        self.system_icc_paths = {
            "sRGB": "/usr/share/color/icc/colord/sRGB.icc",
            "Adobe RGB": "/usr/share/color/icc/colord/AdobeRGB1998.icc",
            "ProPhoto RGB": "/usr/share/color/icc/colord/ProPhotoRGB.icc"
        }
        
        self.init_ui()
        
    def init_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        outer_layout = QVBoxLayout(central_widget)
        
        main_layout = QHBoxLayout()
        outer_layout.addLayout(main_layout, stretch=1)
        
        # View container
        self.stack = QStackedWidget()
        
        # Left Panel - CPU Image View
        self.view = ImageGraphicsView()
        self.scene = QGraphicsScene()
        self.view.setScene(self.scene)
        self.pixmap_item = QGraphicsPixmapItem()
        self.scene.addItem(self.pixmap_item)
        
        # Left Panel - GPU Image View
        self.gl_view = GLImageView()
        
        self.stack.addWidget(self.view)
        self.stack.addWidget(self.gl_view)
        
        main_layout.addWidget(self.stack, stretch=3)
        
        # Right Panel - Controls
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setFixedWidth(300)
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        
        control_panel = QWidget()
        control_layout = QVBoxLayout(control_panel)
        
        scroll_area.setWidget(control_panel)
        main_layout.addWidget(scroll_area)
        
        # Histogram
        self.histogram = HistogramWidget()
        control_layout.addWidget(self.histogram)
        
        # Buttons Grid
        btn_grid = QGridLayout()
        control_layout.addLayout(btn_grid)
        
        btn_open = QPushButton("Open Folder")
        btn_open.clicked.connect(self.open_directory)
        btn_grid.addWidget(btn_open, 0, 0)

        btn_mimic = QPushButton("Match-Camera")
        btn_mimic.clicked.connect(self.match_thumbnail)
        btn_grid.addWidget(btn_mimic, 0, 1)

        btn_reset_view = QPushButton("Reset View")
        btn_reset_view.clicked.connect(self.reset_viewport)
        btn_grid.addWidget(btn_reset_view, 1, 0)

        btn_reset_all = QPushButton("Reset Adjustments")
        btn_reset_all.clicked.connect(self.reset_all_settings)
        btn_grid.addWidget(btn_reset_all, 1, 1)
        
        # Exposure Slider
        self.lbl_exposure = ClickableLabel("Exposure: 0.00 EV")
        self.lbl_exposure.setToolTip("Click to reset")
        self.lbl_exposure.clicked.connect(lambda: self.slider_exposure.setValue(0))
        control_layout.addWidget(self.lbl_exposure)
        self.slider_exposure = QSlider(Qt.Orientation.Horizontal)
        self.slider_exposure.setMinimum(-30) # -5.0 stops
        self.slider_exposure.setMaximum(30)  # +5.0 stops
        self.slider_exposure.setValue(0)     # 0 stops
        self.slider_exposure.valueChanged.connect(self.on_exposure_changed)
        control_layout.addWidget(self.slider_exposure)
        
        # Gamma Slider
        self.lbl_gamma = ClickableLabel("Gamma: 2.22")
        self.lbl_gamma.setToolTip("Click to reset")
        self.lbl_gamma.clicked.connect(lambda: self.slider_gamma.setValue(222))
        control_layout.addWidget(self.lbl_gamma)
        self.slider_gamma = QSlider(Qt.Orientation.Horizontal)
        self.slider_gamma.setMinimum(100) # 0.1
        self.slider_gamma.setMaximum(500) # 5.0
        self.slider_gamma.setValue(222) # 2.22
        self.slider_gamma.valueChanged.connect(self.on_gamma_changed)
        control_layout.addWidget(self.slider_gamma)
        
        # Tonal Controls
        curve_label_layout = QHBoxLayout()
        curve_label_layout.addWidget(QLabel("Tone Curve:"))
        btn_reset_curve = QPushButton("Reset Curve")
        btn_reset_curve.clicked.connect(lambda: self.tone_curve.set_points([(0.0, 0.0), (1.0, 1.0)]))
        curve_label_layout.addWidget(btn_reset_curve)
        control_layout.addLayout(curve_label_layout)
        
        self.tone_curve = ToneCurveWidget()
        self.tone_curve.changed.connect(self.on_curve_changed)
        control_layout.addWidget(self.tone_curve)
        
        tonal_grid = QFormLayout()
        control_layout.addLayout(tonal_grid)
        
        def create_tonal_slider(label_text, default_val, attr_name, min_v=-100, max_v=100):
            lbl = ClickableLabel(f"{label_text}: {default_val:+d}")
            lbl.setToolTip("Click to reset")
            slider = QSlider(Qt.Orientation.Horizontal)
            slider.setRange(min_v, max_v)
            slider.setValue(default_val)
            lbl.clicked.connect(lambda: slider.setValue(default_val))
            
            def on_val_changed(v):
                lbl.setText(f"{label_text}: {v:+d}")
                setattr(self, attr_name, v / 100.0)
                self.update_image_deferred()
            
            slider.valueChanged.connect(on_val_changed)
            tonal_grid.addRow(lbl)
            tonal_grid.addRow(slider)
            setattr(self, attr_name, default_val / 100.0)
            return slider

        self.slider_saturation = create_tonal_slider("Saturation", 0, "saturation")
        self.slider_hl_protect = create_tonal_slider("HL Protect", 0, "hl_protect", 0, 100)
        
        # White Balance
        control_layout.addWidget(QLabel("White Balance:"))
        self.cmb_wb = QComboBox()
        self.cmb_wb.addItems(["Camera", "Auto"])
        self.cmb_wb.currentIndexChanged.connect(self.on_cache_invalidating_change)
        control_layout.addWidget(self.cmb_wb)
        
        # Temperature Offset
        self.lbl_temp = ClickableLabel("Temp Offset: 0 K")
        self.lbl_temp.setToolTip("Click to reset")
        self.lbl_temp.clicked.connect(lambda: self.slider_temp.setValue(0))
        control_layout.addWidget(self.lbl_temp)
        
        temp_layout = QHBoxLayout()
        self.slider_temp = QSlider(Qt.Orientation.Horizontal)
        self.slider_temp.setMinimum(-4000)
        self.slider_temp.setMaximum(4000)
        self.slider_temp.setValue(0)
        self.slider_temp.valueChanged.connect(self.on_wb_slider_changed)
        temp_layout.addWidget(self.slider_temp)
        
        self.spin_temp = QSpinBox()
        self.spin_temp.setRange(-4000, 4000)
        self.spin_temp.setValue(0)
        self.spin_temp.setSuffix(" K")
        self.spin_temp.setFixedWidth(80)
        self.spin_temp.valueChanged.connect(self.on_wb_spin_changed)
        temp_layout.addWidget(self.spin_temp)
        control_layout.addLayout(temp_layout)

        # Tint Offset
        self.lbl_tint = ClickableLabel("Tint Offset: 0")
        self.lbl_tint.setToolTip("Click to reset")
        self.lbl_tint.clicked.connect(lambda: self.slider_tint.setValue(0))
        control_layout.addWidget(self.lbl_tint)
        
        tint_layout = QHBoxLayout()
        self.slider_tint = QSlider(Qt.Orientation.Horizontal)
        self.slider_tint.setMinimum(-100)
        self.slider_tint.setMaximum(100)
        self.slider_tint.setValue(0)
        self.slider_tint.valueChanged.connect(self.on_wb_slider_changed)
        tint_layout.addWidget(self.slider_tint)
        
        self.spin_tint = QSpinBox()
        self.spin_tint.setRange(-100, 100)
        self.spin_tint.setValue(0)
        self.spin_tint.setFixedWidth(80)
        self.spin_tint.valueChanged.connect(self.on_wb_spin_changed)
        tint_layout.addWidget(self.spin_tint)
        control_layout.addLayout(tint_layout)
        
        # Output Color Space
        control_layout.addWidget(QLabel("Output Color Space:"))
        self.cmb_colorspace = QComboBox()
        self.cmb_colorspace.addItems(["sRGB", "Adobe RGB", "ProPhoto RGB"])
        self.cmb_colorspace.currentIndexChanged.connect(self.on_cache_invalidating_change)
        control_layout.addWidget(self.cmb_colorspace)
        
        # Monitor ICC Profile
        self.lbl_profile = QLabel()
        self.lbl_profile.setWordWrap(True)
        self.update_profile_label()
        control_layout.addWidget(self.lbl_profile)
        
        self.chk_disable_monitor_profile = QCheckBox("Disable Monitor Profile")
        self.chk_disable_monitor_profile.setToolTip("Skip converting to the detected monitor ICC profile (targets sRGB instead)")
        self.chk_disable_monitor_profile.stateChanged.connect(self.request_update_image)
        control_layout.addWidget(self.chk_disable_monitor_profile)
        
        btn_settings = QPushButton("Settings")
        btn_settings.clicked.connect(self.open_settings)
        control_layout.addWidget(btn_settings)
        
        # Export
        control_layout.addStretch()
        export_btn_layout = QHBoxLayout()
        btn_export = QPushButton("Export Image")
        btn_export.clicked.connect(self.export_image)
        export_btn_layout.addWidget(btn_export)
        self.btn_export_all = QPushButton("Export All")
        self.btn_export_all.setToolTip("Export all images in the directory with their saved settings")
        self.btn_export_all.clicked.connect(self.export_all_images)
        export_btn_layout.addWidget(self.btn_export_all)
        self.btn_stop_export = QPushButton("Stop Export")
        self.btn_stop_export.setToolTip("Cancel the running batch export")
        self.btn_stop_export.setVisible(False)
        self.btn_stop_export.clicked.connect(self._request_stop_export)
        export_btn_layout.addWidget(self.btn_stop_export)
        control_layout.addLayout(export_btn_layout)
        
        self.timer = QTimer()
        self.timer.setSingleShot(True)
        self.timer.timeout.connect(self.request_update_image)
        
        self.save_timer = QTimer()
        self.save_timer.setSingleShot(True)
        self.save_timer.timeout.connect(self._auto_save_settings)
        
        # Filmstrip
        self.filmstrip = QListWidget()
        self.filmstrip.setViewMode(QListWidget.ViewMode.IconMode)
        self.filmstrip.setIconSize(QSize(120, 80))
        self.filmstrip.setGridSize(QSize(128, 110))
        self.filmstrip.setFlow(QListWidget.Flow.LeftToRight)
        self.filmstrip.setWrapping(False)
        self.filmstrip.setResizeMode(QListWidget.ResizeMode.Adjust)
        self.filmstrip.setFixedHeight(130)
        self.filmstrip.setHorizontalScrollMode(QListWidget.ScrollMode.ScrollPerPixel)
        self.filmstrip.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.filmstrip.currentItemChanged.connect(self.on_thumbnail_selected)
        outer_layout.addWidget(self.filmstrip)

        # Filmstrip footer: file count on the left, clear-settings button on the right
        filmstrip_footer = QHBoxLayout()
        self.lbl_file_count = QLabel("")
        self.lbl_file_count.setStyleSheet("color: gray; font-size: 11px;")
        filmstrip_footer.addWidget(self.lbl_file_count)
        filmstrip_footer.addStretch()
        self.btn_clear_settings = QPushButton("Clear All Saved Settings")
        self.btn_clear_settings.setToolTip("Discard persisted adjustments for every image in the current directory")
        self.btn_clear_settings.setEnabled(False)
        self.btn_clear_settings.clicked.connect(self.clear_all_saved_settings)
        filmstrip_footer.addWidget(self.btn_clear_settings)
        outer_layout.addLayout(filmstrip_footer)
        
    def open_directory(self):
        dir_name = QFileDialog.getExistingDirectory(self, "Open Directory", "")
        if dir_name:
            self.load_directory(dir_name)
            
    def match_thumbnail(self):
        if self.raw_image is None:
            return
            
        try:
            # 1. Get Thumbnail
            thumb = self.raw_image.extract_thumb()
            if thumb.format != rawpy.ThumbFormat.JPEG:
                return
            
            thumb_img = Image.open(io.BytesIO(thumb.data))
            # Downsample to get average color
            thumb_small = thumb_img.resize((32, 32), Image.Resampling.LANCZOS)
            thumb_arr = np.array(thumb_small).astype(np.float32) / 255.0
            
            # Approximate linear space (thumbnail is likely sRGB)
            thumb_lin = np.power(thumb_arr, 2.2)
            r_tgt = np.mean(thumb_lin[:,:,0])
            g_tgt = np.mean(thumb_lin[:,:,1])
            b_tgt = np.mean(thumb_lin[:,:,2])
            
            # 2. Get Raw Linear average
            if self.linear_cache is None:
                self.request_update_image()
            if self.linear_cache is None:
                return
                
            raw_arr = self.linear_cache[::32, ::32, :].astype(np.float32) / 65535.0
            r_src = np.mean(raw_arr[:,:,0])
            g_src = np.mean(raw_arr[:,:,1])
            b_src = np.mean(raw_arr[:,:,2])
            
            # 3. Compute Multipliers
            m_r = r_tgt / max(r_src, 1e-6)
            m_g = g_tgt / max(g_src, 1e-6)
            m_b = b_tgt / max(b_src, 1e-6)
            
            # 4. Set Exposure based on Green channel
            ev = math.log2(max(m_g, 1e-6))
            ev = max(-5.0, min(5.0, ev))
            self.slider_exposure.setValue(int(ev * 6))
            
            # 5. Set White Balance Offsets
            res_r = m_r / max(m_g, 1e-6)
            res_g = 1.0
            res_b = m_b / max(m_g, 1e-6)
            
            r_ref, g_ref, b_ref = self.kelvin_to_rgb(6500)
            target_r_off = r_ref / max(res_r, 1e-6)
            target_b_off = b_ref / max(res_b, 1e-6)
            target_ratio = target_r_off / max(target_b_off, 1e-6)
            
            best_dt = 0
            min_diff = float('inf')
            for dt in range(-4000, 4001, 20):
                k = np.clip(6500 + dt, 2000, 12000)
                r, g, b = self.kelvin_to_rgb(k)
                if b == 0: continue
                ratio = r / b
                diff = abs(ratio - target_ratio)
                if diff < min_diff:
                    min_diff = diff
                    best_dt = dt
                    
            r_k, g_k, b_k = self.kelvin_to_rgb(6500 + best_dt)
            target_g_off = g_ref / max(res_g, 1e-6)
            best_dtint = 200.0 * (1.0 - target_g_off / max(g_k, 1e-6))
            
            # 6. Compute Saturation Match
            def get_chroma(arr):
                luma = 0.299 * arr[:,:,0] + 0.587 * arr[:,:,1] + 0.114 * arr[:,:,2]
                luma = np.maximum(luma, 1e-6)
                chroma = np.abs(arr[:,:,0] - luma) + np.abs(arr[:,:,1] - luma) + np.abs(arr[:,:,2] - luma)
                return np.mean(chroma / luma)
                
            tgt_chroma = get_chroma(thumb_lin)
            src_chroma = get_chroma(raw_arr)
            sat_mult = tgt_chroma / max(src_chroma, 1e-6)
            sat_val = (sat_mult - 1.0) * 100.0
            
            # 7. Tone Curve Matching
            raw_adj = raw_arr * np.array([res_r * max(m_g, 1e-6), max(m_g, 1e-6), res_b * max(m_g, 1e-6)])
            
            luma_tgt = 0.299 * thumb_lin[:,:,0] + 0.587 * thumb_lin[:,:,1] + 0.114 * thumb_lin[:,:,2]
            luma_src = 0.299 * raw_adj[:,:,0] + 0.587 * raw_adj[:,:,1] + 0.114 * raw_adj[:,:,2]
            
            p_levels = np.linspace(5, 95, 7)
            tgt_p = np.percentile(luma_tgt, p_levels)
            src_p = np.percentile(luma_src, p_levels)
            
            points = [(0.0, 0.0)]
            for s, t in zip(src_p, tgt_p):
                s_val = float(np.clip(s, 0.01, 0.99))
                t_val = float(np.clip(t, 0.01, 0.99))
                # Soften the curve by blending 30% towards linear
                t_val = 0.7 * t_val + 0.3 * s_val
                points.append((s_val, t_val))
            points.append((1.0, 1.0))
            
            filtered_points = [points[0]]
            for p in points[1:-1]:
                # Ensure minimum distance in X
                if p[0] > filtered_points[-1][0] + 0.08:
                    dx = p[0] - filtered_points[-1][0]
                    # Constrain slope to avoid excessively steep curves
                    new_y = np.clip(p[1], filtered_points[-1][1] + dx * 0.2, filtered_points[-1][1] + dx * 2.5)
                    filtered_points.append((p[0], new_y))
                    
            if 1.0 > filtered_points[-1][0] + 0.05:
                filtered_points.append((1.0, 1.0))
            else:
                filtered_points[-1] = (1.0, 1.0)
            
            # Auto Highlight Protect calculation
            hl_protect_val = 0
            if src_p[-1] > tgt_p[-1]:
                hl_protect_val = int(np.clip((src_p[-1] - tgt_p[-1]) * 200, 0, 100))
            
            self.tone_curve.set_points(filtered_points)
            self.slider_saturation.setValue(int(np.clip(sat_val, -100, 100)))
            self.slider_gamma.setValue(222)
            self.slider_hl_protect.setValue(hl_protect_val)
            
            self.slider_temp.setValue(int(np.clip(best_dt, -4000, 4000)))
            self.slider_tint.setValue(int(np.clip(best_dtint, -100, 100)))
            
        except Exception as e:
            QMessageBox.warning(self, "Warning", f"Could not match thumbnail: {e}")
            
    def load_directory(self, dir_name):
        self.filmstrip.clear()
        self.image_files = []
        self.current_dir = dir_name
        valid_extensions = ('.cr2', '.cr3', '.nef', '.arw', '.dng', '.pef')
        try:
            for f in sorted(os.listdir(dir_name)):
                if f.lower().endswith(valid_extensions):
                    full_path = os.path.join(dir_name, f)
                    self.image_files.append(full_path)

                    item = QListWidgetItem(f)
                    self.filmstrip.addItem(item)

            # Load persisted settings for this directory
            self.image_settings = self._load_settings_from_disk(dir_name)

            total = len(self.image_files)
            if total > 0:
                self.lbl_file_count.setText(f"{total} file{'s' if total != 1 else ''} in directory")
                self.btn_clear_settings.setEnabled(True)
            else:
                self.lbl_file_count.setText("No RAW files found")
                self.btn_clear_settings.setEnabled(False)

            # Process thumbnails asynchronously using true multiprocessing
            self.thumbnail_manager = ThumbnailManager(self.image_files)
            self.thumbnail_manager.signals.result.connect(self.on_thumbnail_ready)
            self.thumbnail_pool.start(self.thumbnail_manager)

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load directory: {str(e)}")

    def on_thumbnail_ready(self, index, data):
        if index < self.filmstrip.count():
            img = QImage()
            img.loadFromData(data)
            base_pixmap = QPixmap.fromImage(img)
            item = self.filmstrip.item(index)
            # Store the raw thumbnail so we can re-badge it later
            item.setData(Qt.ItemDataRole.UserRole, base_pixmap)
            if index < len(self.image_files) and self._is_edited(self.image_files[index]):
                item.setIcon(QIcon(self._make_badge_icon(base_pixmap)))
            else:
                item.setIcon(QIcon(base_pixmap))

    def on_thumbnail_selected(self, current, previous=None):
        if current is None:
            return
        row = self.filmstrip.row(current)
        if 0 <= row < len(self.image_files):
            if self.image_files[row] != self.raw_path:
                self.load_image(self.image_files[row])

    def _auto_save_settings(self):
        if self.raw_path is not None:
            self.image_settings[self.raw_path] = self._collect_settings()
            self._save_settings_to_disk()
            if self.raw_path in self.image_files:
                self._refresh_badge_for_index(self.image_files.index(self.raw_path))

    def load_image(self, file_path):
        # --- Save settings for the image we are leaving ---
        self._auto_save_settings()

        self.raw_path = file_path
        try:
            if self.raw_image is not None:
                self.raw_image.close()
            self.raw_image = rawpy.imread(self.raw_path)

            # --- Restore or default settings for the new image ---
            if file_path in self.image_settings:
                self._apply_settings(self.image_settings[file_path])
            else:
                # Default colorspace via EXIF; all other controls to defaults
                default_cs = self.get_default_colorspace(self.raw_path)
                self._apply_settings(self._default_settings(default_cs))

            self.cache_dirty = True
            self.is_first_load = True
            self.request_update_image()
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load image: {str(e)}")
                
    def get_default_colorspace(self, file_path):
        color_space = "sRGB"
        if os.path.basename(file_path).startswith('_'):
            color_space = "Adobe RGB"
        else:
            try:
                with open(file_path, 'rb') as f:
                    tags = exifread.process_file(f, details=False)
                    if 'EXIF ColorSpace' in tags:
                        val = str(tags['EXIF ColorSpace'])
                        if 'Uncalibrated' in val or '65535' in val or 'Adobe' in val or '2' in val:
                            color_space = "Adobe RGB"
            except Exception:
                pass
        return color_space

    def reset_viewport(self):
        if self.processing_mode == 3:
            self.gl_view.fit_in_view()
        else:
            self.view.fitInView(self.scene.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)

    def reset_all_settings(self):
        # Reset dropdowns
        self.cmb_wb.setCurrentIndex(0) # Camera
        
        # Restore default colorspace for the loaded image
        if self.raw_path:
            self.cmb_colorspace.setCurrentText(self.get_default_colorspace(self.raw_path))
        else:
            self.cmb_colorspace.setCurrentIndex(0) # sRGB default

        # Set sliders to defaults
        self.slider_exposure.setValue(0)
        self.on_exposure_changed(0) # Force label update
        
        self.slider_gamma.setValue(222)
        self.on_gamma_changed(222) # Force label update
        
        self.tone_curve.set_points([(0.0, 0.0), (1.0, 1.0)])
        self.slider_saturation.setValue(0)
        self.slider_hl_protect.setValue(0)
        
        self.slider_temp.setValue(0)
        self.slider_tint.setValue(0)
        self.on_wb_slider_changed() # Force labels and spin boxes update
        
        self.request_update_image()

    # ------------------------------------------------------------------
    # Per-image settings helpers
    # ------------------------------------------------------------------
    def _default_settings(self, colorspace=None):
        """Return a dict representing the application's default adjustments."""
        return {
            'wb':         'Camera',
            'colorspace': colorspace or 'sRGB',
            'exposure':   0,
            'gamma':      222,
            'tone_curve': [[0.0, 0.0], [1.0, 1.0]],
            'saturation': 0,
            'hl_protect': 0,
            'temp':       0,
            'tint':       0,
        }

    def _collect_settings(self):
        """Snapshot every adjustable control into a plain dict."""
        return {
            'wb':         self.cmb_wb.currentText(),
            'colorspace': self.cmb_colorspace.currentText(),
            'exposure':   self.slider_exposure.value(),
            'gamma':      self.slider_gamma.value(),
            'tone_curve': self.tone_curve.get_points(),
            'saturation': self.slider_saturation.value(),
            'hl_protect': self.slider_hl_protect.value(),
            'temp':       self.slider_temp.value(),
            'tint':       self.slider_tint.value(),
        }

    def _apply_settings(self, s):
        """Restore all controls from a settings dict without triggering extra redraws."""
        # Block signals on combos to avoid double cache invalidation
        self.cmb_wb.blockSignals(True)
        self.cmb_colorspace.blockSignals(True)

        self.cmb_wb.setCurrentText(s['wb'])
        self.cmb_colorspace.setCurrentText(s['colorspace'])

        self.cmb_wb.blockSignals(False)
        self.cmb_colorspace.blockSignals(False)

        # Sliders: let their valueChanged handlers run normally so labels update
        self.slider_exposure.setValue(s['exposure'])
        self.slider_gamma.setValue(s['gamma'])
        self.tone_curve.set_points(s.get('tone_curve', [[0.0, 0.0], [1.0, 1.0]]))
        self.slider_saturation.setValue(s['saturation'])
        self.slider_hl_protect.setValue(s.get('hl_protect', 0))
        self.slider_temp.setValue(s['temp'])
        self.slider_tint.setValue(s['tint'])
        # Sync the spin boxes (slider handlers don't always update spins)
        self.spin_temp.blockSignals(True)
        self.spin_tint.blockSignals(True)
        self.spin_temp.setValue(s['temp'])
        self.spin_tint.setValue(s['tint'])
        self.spin_temp.blockSignals(False)
        self.spin_tint.blockSignals(False)

    # ------------------------------------------------------------------
    # Persistence helpers
    # ------------------------------------------------------------------
    def _settings_file_path(self, dir_name=None):
        d = dir_name or self.current_dir
        if not d:
            return None
        return os.path.join(d, self.SETTINGS_FILENAME)

    def _load_settings_from_disk(self, dir_name):
        path = self._settings_file_path(dir_name)
        if path and os.path.isfile(path):
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception:
                pass
        return {}

    def _save_settings_to_disk(self):
        path = self._settings_file_path()
        if not path:
            return
        try:
            # Only write entries that differ from default (edited images)
            to_save = {
                k: v for k, v in self.image_settings.items()
                if self._is_edited(k)
            }
            with open(path, 'w', encoding='utf-8') as f:
                json.dump(to_save, f, indent=2)
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Badge helpers
    # ------------------------------------------------------------------
    def _is_edited(self, file_path):
        """Return True if the stored settings for file_path differ from defaults."""
        if file_path not in self.image_settings:
            return False
        stored = self.image_settings[file_path]
        # Use the stored colorspace as the baseline (EXIF-detected default)
        defaults = self._default_settings(stored.get('colorspace', 'sRGB'))
        return stored != defaults

    def _make_badge_icon(self, base_pixmap):
        """Return a copy of base_pixmap with a small coloured edit-badge overlay."""
        result = base_pixmap.copy()
        painter = QPainter(result)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        # Badge background
        badge_size = max(18, result.height() // 5)
        margin = 4
        bx = result.width() - badge_size - margin
        by = margin
        painter.setBrush(QColor(255, 165, 0, 220))   # orange
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawEllipse(bx, by, badge_size, badge_size)
        # Pencil glyph
        painter.setPen(QColor(255, 255, 255))
        font = painter.font()
        font.setPixelSize(max(10, badge_size - 6))
        font.setBold(True)
        painter.setFont(font)
        painter.drawText(bx, by, badge_size, badge_size,
                         Qt.AlignmentFlag.AlignCenter, "✎")
        painter.end()
        return result

    def _refresh_badge_for_index(self, index):
        """Redraw the badge (or remove it) for a single filmstrip item."""
        if index < 0 or index >= self.filmstrip.count():
            return
        item = self.filmstrip.item(index)
        base_pixmap = item.data(Qt.ItemDataRole.UserRole)
        if base_pixmap is None:
            return   # thumbnail not yet loaded
        if index < len(self.image_files) and self._is_edited(self.image_files[index]):
            item.setIcon(QIcon(self._make_badge_icon(base_pixmap)))
        else:
            item.setIcon(QIcon(base_pixmap))

    def _refresh_all_badges(self):
        for i in range(self.filmstrip.count()):
            self._refresh_badge_for_index(i)

    # ------------------------------------------------------------------
    # Clear all saved settings
    # ------------------------------------------------------------------
    def clear_all_saved_settings(self):
        reply = QMessageBox.question(
            self, "Clear All Saved Settings",
            "This will permanently discard saved adjustments for every image "
            "in the current directory. Continue?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )
        if reply != QMessageBox.StandardButton.Yes:
            return

        self.image_settings.clear()

        # Delete the JSON sidecar on disk
        path = self._settings_file_path()
        if path and os.path.isfile(path):
            try:
                os.remove(path)
            except Exception:
                pass

        # Remove all badges from thumbnails
        self._refresh_all_badges()

        # If an image is currently open, revert it to defaults
        if self.raw_path:
            default_cs = self.get_default_colorspace(self.raw_path)
            self._apply_settings(self._default_settings(default_cs))
            self.cache_dirty = True
            self.request_update_image()

    def open_settings(self):
        old_mode = self.processing_mode
        old_profile = self.custom_profile_path
        old_fast_preview = self.fast_preview
        
        dlg = SettingsDialog(self)
        if dlg.exec():
            self.processing_mode = dlg.cmb_mode.currentIndex()
            self.fast_preview = dlg.chk_fast_preview.isChecked()
            self.update_profile_label()
            
            if old_mode != self.processing_mode or old_profile != self.custom_profile_path or old_fast_preview != self.fast_preview:
                if self.processing_mode > 0 and self.linear_cache is None:
                    self.cache_dirty = True
                if old_fast_preview != self.fast_preview:
                    self.cache_dirty = True
                self.request_update_image()

    def update_profile_label(self):
        if self.custom_profile_path:
            text = f"Monitor Profile: Custom ({os.path.basename(self.custom_profile_path)})"
        elif self.monitor_profile_bytes:
            text = "Monitor Profile: Auto-detected (X11)"
        elif self.monitor_profile_path:
            text = f"Monitor Profile: Auto-detected ({os.path.basename(self.monitor_profile_path)})"
        else:
            text = "Monitor Profile: None detected (Fallback to sRGB)"
        self.lbl_profile.setText(text)

    def on_cache_invalidating_change(self):
        self.cache_dirty = True
        self.request_update_image()

    def on_wb_slider_changed(self):
        temp = self.slider_temp.value()
        tint = self.slider_tint.value()
        
        # Sync spin boxes
        self.spin_temp.blockSignals(True)
        self.spin_tint.blockSignals(True)
        self.spin_temp.setValue(temp)
        self.spin_tint.setValue(tint)
        self.spin_temp.blockSignals(False)
        self.spin_tint.blockSignals(False)
        
        self.lbl_temp.setText(f"Temp Offset: {temp:+d} K")
        self.lbl_tint.setText(f"Tint Offset: {tint:+d}")
        self.update_image_deferred()

    def on_wb_spin_changed(self):
        temp = self.spin_temp.value()
        tint = self.spin_tint.value()
        
        # Sync sliders
        self.slider_temp.blockSignals(True)
        self.slider_tint.blockSignals(True)
        self.slider_temp.setValue(temp)
        self.slider_tint.setValue(tint)
        self.slider_temp.blockSignals(False)
        self.slider_tint.blockSignals(False)
        
        self.lbl_temp.setText(f"Temp Offset: {temp:+d} K")
        self.lbl_tint.setText(f"Tint Offset: {tint:+d}")
        self.update_image_deferred()

    def kelvin_to_rgb(self, kelvin):
        temp = kelvin / 100.0
        if temp <= 66:
            r = 255
            g = temp
            g = 99.4708025861 * math.log(g) - 161.1195681661
            if temp <= 19:
                b = 0
            else:
                b = temp - 10
                b = 138.5177312231 * math.log(b) - 305.0447927307
        else:
            r = temp - 60
            r = 329.698727446 * math.pow(r, -0.1332047592)
            g = temp - 60
            g = 288.1221695283 * math.pow(g, -0.0755148492)
            b = 255
        
        return np.clip(r, 0, 255), np.clip(g, 0, 255), np.clip(b, 0, 255)

    def calculate_wb_offsets(self):
        dt = self.slider_temp.value()
        dtint = self.slider_tint.value()
        
        # Target color temperature shift relative to a neutral 6500K point
        r_ref, g_ref, b_ref = self.kelvin_to_rgb(6500)
        # Shifted color (clamped to supported range 2000-12000)
        r_off, g_off, b_off = self.kelvin_to_rgb(np.clip(6500 + dt, 2000, 12000))
        
        # Apply tint to green
        g_off = g_off * (1.0 - dtint / 200.0)
        
        # Calculate multipliers needed to reach the offset color from neutral
        s_r = r_ref / r_off
        s_g = g_ref / g_off
        s_b = b_ref / b_off
        
        return [s_r, s_g, s_b]

    def multipliers_to_kelvin_tint(self, multipliers):
        # multipliers is [r, g1, b, g2]
        if not multipliers or len(multipliers) < 3:
            return 6500, 0
            
        # Reference daylight balance
        ref_mults = getattr(self.raw_image, 'daylight_whitebalance', [2.0, 1.0, 1.5, 1.0])
        if all(m == 0 for m in ref_mults[:3]):
            ref_mults = [2.0, 1.0, 1.5, 1.0]
        ref_r, ref_g, ref_b = self.kelvin_to_rgb(5500)
        
        # Back-calculate the actual light color from the multipliers and sensor sensitivity
        # color = (ref_mults * ref_color) / multipliers
        light_r = (ref_mults[0] * ref_r) / multipliers[0]
        light_b = (ref_mults[2] * ref_b) / multipliers[2]
        light_g = (ref_mults[1] * ref_g) / multipliers[1]
        
        target_rb_ratio = light_r / light_b
        
        best_kelvin = 6500
        min_diff = float('inf')
        for k in range(2000, 12001, 50):
            r, g, b = self.kelvin_to_rgb(k)
            rb_ratio = r / b
            diff = abs(rb_ratio - target_rb_ratio)
            if diff < min_diff:
                min_diff = diff
                best_kelvin = k
        
        # Back-calculate tint from the green channel
        r_k, g_k, b_k = self.kelvin_to_rgb(best_kelvin)
        # light_g = g_k * (1 - tint/200)
        tint = 200 * (1.0 - (light_g / g_k))
        tint = int(np.clip(tint, -100, 100))
        
        return best_kelvin, tint

    def update_wb_sliders_from_mode(self):
        pass # Offsets are always active, no need to sync absolute values now

    def on_exposure_changed(self, value):
        ev = value / 6.0
        self.lbl_exposure.setText(f"Exposure: {ev:+.2f} EV")
        self.update_image_deferred()

    def on_gamma_changed(self, value):
        gamma_val = value / 100.0
        self.lbl_gamma.setText(f"Gamma: {gamma_val:.2f}")
        self.update_image_deferred()

    def on_curve_changed(self):
        if self.processing_mode == 3:
            self.gl_view.set_curve(self.tone_curve.get_points())
        self.update_image_deferred()

    def update_image_deferred(self):
        if self.raw_image is None:
            return
            
        # Start or restart autosave timer
        self.save_timer.start(1000)
        
        if self.processing_mode == 3:
            # GPU Mode can update instantly without debounce lag
            self.request_update_image()
        else:
            self.timer.start(100) # 100ms debounce for CPU modes

    def request_update_image(self):
        if self.raw_image is None:
            return
        
        # Don't show wait cursor for instant GPU shader updates
        if self.processing_mode != 3:
            QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
            
        try:
            self.process_and_display()
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to process image: {str(e)}")
        finally:
            if self.processing_mode != 3:
                QApplication.restoreOverrideCursor()

    def get_display_profile(self):
        if self.chk_disable_monitor_profile.isChecked():
            return ImageCms.createProfile("sRGB")
        if self.custom_profile_path:
            return ImageCms.ImageCmsProfile(self.custom_profile_path)
        if self.monitor_profile_bytes:
            return ImageCms.ImageCmsProfile(io.BytesIO(self.monitor_profile_bytes))
        if self.monitor_profile_path:
            return ImageCms.ImageCmsProfile(self.monitor_profile_path)
        return ImageCms.createProfile("sRGB")

    def apply_tonal_math(self, arr):
        pts = self.tone_curve.get_points()
        lut = interpolate_curve(pts, 65536)
        x_idx = np.clip(arr * 65535, 0, 65535).astype(np.int32)
        return lut[x_idx]

    def update_gpu_lut(self, cs_text, gamma_val):
        lut_key = (cs_text, self.chk_disable_monitor_profile.isChecked())
        if not hasattr(self, '_gpu_lut_cache'):
            self._gpu_lut_cache = {}
            
        if lut_key not in self._gpu_lut_cache:
            lut_size = 33
            target_icc_path = self.system_icc_paths.get(cs_text)
            if target_icc_path and os.path.exists(target_icc_path):
                source_profile = ImageCms.ImageCmsProfile(target_icc_path)
            else:
                source_profile = ImageCms.createProfile("sRGB")
                
            display_profile = self.get_display_profile()
            try:
                transform = ImageCms.buildTransform(source_profile, display_profile, "RGB", "RGB", ImageCms.Intent.PERCEPTUAL, 0)
                
                x = np.linspace(0, 255, lut_size, dtype=np.uint8)
                z, y, x = np.meshgrid(x, x, x, indexing='ij')
                lut_data = np.stack([x, y, z], axis=-1)
                
                flat_rgb = lut_data.reshape(-1, 3)
                img = Image.fromarray(flat_rgb.reshape(-1, lut_size, 3))
                
                ImageCms.applyTransform(img, transform, inPlace=True)
                self._gpu_lut_cache[lut_key] = np.array(img).reshape(lut_size, lut_size, lut_size, 3)
            except Exception as e:
                print("Failed to build GPU LUT:", e)
                x = np.linspace(0, 255, lut_size, dtype=np.uint8)
                z, y, x = np.meshgrid(x, x, x, indexing='ij')
                self._gpu_lut_cache[lut_key] = np.stack([x, y, z], axis=-1)
            
        lut_data_out = self._gpu_lut_cache[lut_key]
        
        if cs_text == 'sRGB': base_gamma = 2.22
        elif cs_text == 'Adobe RGB': base_gamma = 2.2
        else: base_gamma = 1.8
        
        encode_power = (2.22 / gamma_val) / base_gamma
        self.gl_view.set_color_lut(lut_data_out, encode_power)


    def process_and_display(self):
        ev = self.slider_exposure.value() / 6.0
        exposure_multiplier = 2.0 ** ev
        gamma_val = self.slider_gamma.value() / 100.0
        
        off = self.calculate_wb_offsets()
        
        use_auto_wb = self.cmb_wb.currentText() == "Auto"
        use_camera_wb = self.cmb_wb.currentText() == "Camera"
        
        cs_text = self.cmb_colorspace.currentText()
        if cs_text == "sRGB":
            out_cs = rawpy.ColorSpace.sRGB
        elif cs_text == "Adobe RGB":
            out_cs = rawpy.ColorSpace.Adobe
        else:
            out_cs = rawpy.ColorSpace.ProPhoto
        half_size = self.fast_preview
        
        # === GPU Shader Mode ===
        if self.processing_mode == 3:
            self.stack.setCurrentWidget(self.gl_view)
            
            wb_mode = self.cmb_wb.currentText()
            wb_offsets = self.calculate_wb_offsets()
            self.gl_view.wb_mults = wb_offsets
            
            if self.cache_dirty or self.linear_cache is None:
                QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
                self.linear_cache = self.raw_image.postprocess(
                    use_camera_wb=(wb_mode == "Camera"),
                    use_auto_wb=(wb_mode == "Auto"),
                    half_size=half_size,
                    output_color=out_cs,
                    output_bps=16,
                    gamma=(1, 1),
                    no_auto_bright=False
                )
                self.cache_dirty = False
                h, w, _ = self.linear_cache.shape
                self.gl_view.set_image(self.linear_cache, w, h)
                self.update_16bit_hist_counts()
                QApplication.restoreOverrideCursor()
                
            if self.is_first_load:
                self.gl_view.fit_in_view()
                self.is_first_load = False
                
            self.update_gpu_lut(cs_text, gamma_val)
            self.gl_view.exposure = exposure_multiplier
            self.gl_view.gamma = gamma_val
            self.gl_view.saturation = getattr(self, 'saturation', 0.0)
            self.gl_view.hl_protect = getattr(self, 'hl_protect', 0.0)
            self.gl_view.set_curve(self.tone_curve.get_points())
            self.gl_view.update()
            
            self.update_display_histogram()
            self.target_icc_path = self.system_icc_paths.get(cs_text)
            return

        # === CPU Modes ===
        self.stack.setCurrentWidget(self.view)
        
        wb_mode = self.cmb_wb.currentText()
        if self.processing_mode == 0:
            # Full rawpy processing (Slow)
            # Full processing doesn't support the offset model easily without custom multipliers
            # so we let rawpy do the base WB and then we'd have to apply offsets after.
            # For simplicity, we just use rawpy's base.
            self.processed_rgb = self.raw_image.postprocess(
                use_camera_wb=(wb_mode == "Camera"),
                use_auto_wb=(wb_mode == "Auto"),
                half_size=half_size,
                exp_shift=exposure_multiplier,
                gamma=(gamma_val, 4.5),
                output_color=out_cs,
                output_bps=8
            )
        else:
            # Mode 1 or 2: use linear cache
            if self.cache_dirty or self.linear_cache is None:
                self.linear_cache = self.raw_image.postprocess(
                    use_camera_wb=(wb_mode == "Camera"),
                    use_auto_wb=(wb_mode == "Auto"),
                    half_size=half_size,
                    output_color=out_cs,
                    output_bps=16,
                    gamma=(1, 1),
                    no_auto_bright=False
                )
                self.cache_dirty = False
                self.update_16bit_hist_counts()
            
            # Apply WB offsets in CPU math path
            off = self.calculate_wb_offsets()
            
            if cs_text == 'sRGB':
                base_gamma = 2.22
            elif cs_text == 'Adobe RGB':
                base_gamma = 2.2
            else:
                base_gamma = 1.8
                
            creative_gamma_adjustment = 2.22 / gamma_val
            encode_power = creative_gamma_adjustment / base_gamma
            
            sat_mult = 1.0 + getattr(self, 'saturation', 0.0)
            hl_protect = getattr(self, 'hl_protect', 0.0)

            if self.processing_mode == 1:
                # Linear Cache Math
                arr = self.linear_cache.astype(np.float32) / 65535.0
                arr[:,:,0] *= off[0]
                arr[:,:,1] *= off[1]
                arr[:,:,2] *= off[2]
                arr = arr * exposure_multiplier
                arr = self.apply_tonal_math(arr)
                
                if sat_mult != 1.0 or hl_protect > 0.0:
                    arr = apply_saturation_and_hl(arr, sat_mult, hl_protect)
                    
                arr = np.power(np.clip(arr, 1e-6, 1.0), encode_power) * 255.0
                self.processed_rgb = arr.astype(np.uint8)
            else:
                # LUT Optimization
                lut = np.arange(65536, dtype=np.float32) / 65535.0
                lut = lut * exposure_multiplier
                lut = self.apply_tonal_math(lut)
                
                res = lut[self.linear_cache]
                res[:,:,0] *= off[0]
                res[:,:,1] *= off[1]
                res[:,:,2] *= off[2]
                res = np.clip(res, 0.0, 1.0)
                
                if sat_mult != 1.0 or hl_protect > 0.0:
                    res = apply_saturation_and_hl(res, sat_mult, hl_protect)
                    
                res = np.power(np.clip(res, 1e-6, 1.0), encode_power) * 255.0
                self.processed_rgb = res.astype(np.uint8)
        
        # Apply Color Management for Display
        img = Image.fromarray(self.processed_rgb)
        
        # Target ICC file path to be used later for embedding in export
        self.target_icc_path = self.system_icc_paths.get(cs_text)
        
        source_profile = None
        if self.target_icc_path and os.path.exists(self.target_icc_path):
            source_profile = ImageCms.ImageCmsProfile(self.target_icc_path)
        else:
            # Fallback if system profile is not found
            if cs_text == "sRGB":
                source_profile = ImageCms.createProfile("sRGB")
            
            if source_profile is None:
                source_profile = ImageCms.createProfile("sRGB")

        display_profile = self.get_display_profile()
        
        try:
            # Create transform from the processed image's color space to the display's color space
            transform = ImageCms.buildTransform(source_profile, display_profile, "RGB", "RGB")
            img = ImageCms.applyTransform(img, transform)
        except Exception as e:
            print("Color management error (fallback to raw output):", e)
            
        # Convert to QPixmap and display
        pixmap = pil2pixmap(img)
        self.pixmap_item.setPixmap(pixmap)
        self.scene.setSceneRect(self.pixmap_item.boundingRect())
        
        # Fit in view only on first load
        if self.is_first_load:
            self.view.fitInView(self.scene.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)
            self.is_first_load = False
        
        self.update_display_histogram()

    def update_16bit_hist_counts(self):
        if self.linear_cache is None: return
        self.counts_r = np.bincount(self.linear_cache[:,:,0].ravel(), minlength=65536)
        self.counts_g = np.bincount(self.linear_cache[:,:,1].ravel(), minlength=65536)
        self.counts_b = np.bincount(self.linear_cache[:,:,2].ravel(), minlength=65536)

    def update_display_histogram(self):
        if self.counts_r is None: return
        
        # Use the same math as the Shader/LUT
        ev = self.slider_exposure.value() / 6.0
        exposure_multiplier = 2.0 ** ev
        gamma_val = self.slider_gamma.value() / 100.0
        
        wb_offsets = self.calculate_wb_offsets()
        
        # Create a mapping from 0-65535 to 0-255
        lut_indices = np.arange(65536, dtype=np.float32) / 65535.0
        
        def apply_to_lut(indices, wb_idx):
            res = indices * wb_offsets[wb_idx] * exposure_multiplier
            res = self.apply_tonal_math(res) # <---
            res = np.power(res, 1.0 / gamma_val) * 255.0
            return res.astype(np.uint8)

        lut_r = apply_to_lut(lut_indices, 0)
        lut_g = apply_to_lut(lut_indices, 1)
        lut_b = apply_to_lut(lut_indices, 2)
        
        # Map 16-bit counts to 8-bit bins
        hist_r = np.bincount(lut_r, weights=self.counts_r, minlength=256)
        hist_g = np.bincount(lut_g, weights=self.counts_g, minlength=256)
        hist_b = np.bincount(lut_b, weights=self.counts_b, minlength=256)
        
        self.histogram.update_hist(hist_r, hist_g, hist_b)

    def export_image(self):
        if self.raw_image is None:
            return
            
        file_name, _ = QFileDialog.getSaveFileName(self, "Export Image", "exported_image.jpg", 
                                                   "JPEG (*.jpg *.jpeg);;TIFF (*.tiff *.tif);;PNG (*.png)")
        if not file_name:
            return
            
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            # Reprocess at full size
            ev = self.slider_exposure.value() / 6.0
            exposure_multiplier = 2.0 ** ev
            gamma_val = self.slider_gamma.value() / 100.0
            
            use_auto_wb = self.cmb_wb.currentText() == "Auto"
            use_camera_wb = self.cmb_wb.currentText() == "Camera"
            
            cs_text = self.cmb_colorspace.currentText()
            if cs_text == "sRGB":
                out_cs = rawpy.ColorSpace.sRGB
                base_gamma = 2.22
            elif cs_text == "Adobe RGB":
                out_cs = rawpy.ColorSpace.Adobe
                base_gamma = 2.2
            else:
                out_cs = rawpy.ColorSpace.ProPhoto
                base_gamma = 1.8
                
            # Process to 16-bit linear for high-quality math
            export_lin = self.raw_image.postprocess(
                use_camera_wb=use_camera_wb,
                use_auto_wb=use_auto_wb,
                half_size=False,
                exp_shift=1.0,
                gamma=(1, 1),
                output_color=out_cs,
                output_bps=16,
                no_auto_bright=True
            )
            
            # Apply WB offsets and Tonal adjustments in 16-bit space
            off = self.calculate_wb_offsets()
            arr = export_lin.astype(np.float32) / 65535.0
            arr[:,:,0] *= off[0] * exposure_multiplier
            arr[:,:,1] *= off[1] * exposure_multiplier
            arr[:,:,2] *= off[2] * exposure_multiplier
            
            arr = self.apply_tonal_math(arr)
            
            sat_mult = 1.0 + getattr(self, 'saturation', 0.0)
            hl_protect = getattr(self, 'hl_protect', 0.0)
            
            if sat_mult != 1.0 or hl_protect > 0.0:
                arr = apply_saturation_and_hl(arr, sat_mult, hl_protect)
            
            # Apply Gamma
            creative_gamma_adjustment = 2.22 / gamma_val
            encode_power = creative_gamma_adjustment / base_gamma
            arr = np.power(np.clip(arr, 1e-6, 1.0), encode_power) * 255.0
            export_rgb = arr.astype(np.uint8)
            
            img = Image.fromarray(export_rgb)
            
            # Embed ICC Profile
            icc_bytes = None
            if self.target_icc_path and os.path.exists(self.target_icc_path):
                with open(self.target_icc_path, "rb") as f:
                    icc_bytes = f.read()
            elif cs_text == "sRGB":
                prof = ImageCms.createProfile("sRGB")
                icc_bytes = prof.tobytes()
                
            if icc_bytes:
                img.save(file_name, icc_profile=icc_bytes)
                QMessageBox.information(self, "Success", f"Image successfully exported to:\n{file_name}")
            else:
                img.save(file_name)
                QMessageBox.warning(self, "Warning", f"Image exported to:\n{file_name}\n\nWarning: Missing system ICC profile. No color profile was embedded!")
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to export image: {str(e)}")
        finally:
            QApplication.restoreOverrideCursor()
    def export_all_images(self):
        """Batch-export all images using ProcessPoolExecutor — UI stays responsive."""
        if not self.image_files:
            QMessageBox.warning(self, "Export All", "No directory is open.")
            return

        # Save the current image's settings before exporting
        if self.raw_path is not None:
            self.image_settings[self.raw_path] = self._collect_settings()
            self._save_settings_to_disk()

        # Format picker dialog
        from PyQt6.QtWidgets import QDialogButtonBox as _DBB
        fmt_dlg = QDialog(self)
        fmt_dlg.setWindowTitle("Export All — Options")
        vbox = QVBoxLayout(fmt_dlg)
        vbox.addWidget(QLabel("Export format:"))
        fmt_combo = QComboBox()
        fmt_combo.addItems(["JPEG", "TIFF", "PNG"])
        vbox.addWidget(fmt_combo)
        btns = _DBB(_DBB.StandardButton.Ok | _DBB.StandardButton.Cancel)
        btns.accepted.connect(fmt_dlg.accept)
        btns.rejected.connect(fmt_dlg.reject)
        vbox.addWidget(btns)
        if fmt_dlg.exec() != QDialog.DialogCode.Accepted:
            return

        fmt_map = {"JPEG": ("jpg", "JPEG"), "TIFF": ("tif", "TIFF"), "PNG": ("png", "PNG")}
        ext, pil_fmt = fmt_map[fmt_combo.currentText()]

        out_dir = QFileDialog.getExistingDirectory(self, "Select Output Folder", self.current_dir or "")
        if not out_dir:
            return

        # Build task list
        tasks   = []
        skipped = []
        for file_path in self.image_files:
            if file_path in self.image_settings:
                s = self.image_settings[file_path]
            else:
                s = self._default_settings(self.get_default_colorspace(file_path))
                skipped.append(os.path.basename(file_path))
            out_path = os.path.join(
                out_dir,
                os.path.splitext(os.path.basename(file_path))[0] + "." + ext
            )
            tasks.append((file_path, s, out_path, pil_fmt, self.system_icc_paths))

        total = len(tasks)

        # Show Stop button, hide Export All, show live progress
        self._export_stop_event = threading.Event()
        self.btn_export_all.setVisible(False)
        self.btn_stop_export.setVisible(True)
        self.lbl_file_count.setText(f"Exporting 0 / {total}…")

        worker = BatchExportWorker(tasks, skipped, self._export_stop_event)
        worker.signals.progress.connect(self._on_export_progress)
        worker.signals.finished.connect(
            lambda sk, fa, dn: self._on_export_finished(sk, fa, dn, total, out_dir)
        )
        self.thumbnail_pool.start(worker)

    def _request_stop_export(self):
        if hasattr(self, '_export_stop_event'):
            self._export_stop_event.set()
        self.btn_stop_export.setEnabled(False)
        self.btn_stop_export.setText("Stopping…")

    def _on_export_progress(self, done, total, workers):
        self.lbl_file_count.setText(f"Exporting {done} / {total}  —  {workers} worker{'s' if workers != 1 else ''} active…")

    def _on_export_finished(self, skipped, failed, done, total, out_dir):
        # Restore UI
        n = len(self.image_files)
        self.lbl_file_count.setText(f"{n} file{'s' if n != 1 else ''} in directory")
        self.btn_stop_export.setVisible(False)
        self.btn_stop_export.setEnabled(True)
        self.btn_stop_export.setText("Stop Export")
        self.btn_export_all.setVisible(True)

        cancelled = hasattr(self, '_export_stop_event') and self._export_stop_event.is_set()

        lines = [f"Exported {done - len(failed)} / {total} images to:\n{out_dir}"]
        if skipped:
            lines.append(f"\n{len(skipped)} image(s) used default settings (never opened):")
            lines.append("  " + ", ".join(skipped[:5]) + ("..." if len(skipped) > 5 else ""))
        if failed:
            lines.append(f"\n{len(failed)} image(s) failed:")
            lines.append("\n".join(failed[:5]))
        if cancelled:
            lines.insert(0, "Export cancelled early.")
        QMessageBox.information(self, "Export All — Complete", "\n".join(lines))


if __name__ == "__main__":
    app = QApplication(sys.argv)
    
    # Make labels and buttons 1 point smaller
    font = app.font()
    if font.pointSize() > 0:
        font.setPointSize(max(1, font.pointSize() - 1))
    elif font.pixelSize() > 0:
        font.setPixelSize(max(1, font.pixelSize() - 1))
        
    app.setFont(font, "QLabel")
    app.setFont(font, "QPushButton")
    app.setFont(font, "QCheckBox")
    app.setFont(font, "QSpinBox")
    
    editor = RawEditor()
    editor.show()
    sys.exit(app.exec())
