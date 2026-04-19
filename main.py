import sys
import os
import subprocess
import io
import time
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QPushButton, QLabel, QFileDialog, 
                             QSlider, QComboBox, QCheckBox, QGraphicsView, 
                             QGraphicsScene, QGraphicsPixmapItem, QMessageBox,
                             QDialog, QFormLayout, QStackedWidget)
from PyQt6.QtOpenGLWidgets import QOpenGLWidget
from PyQt6.QtCore import Qt, QTimer, pyqtSignal
from PyQt6.QtGui import QPixmap, QImage, QPainter, QColor, QPainterPath
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
        self.shader_program = shaders.compileProgram(
            shaders.compileShader("""
                #version 330 core
                layout (location = 0) in vec2 aPos;
                layout (location = 1) in vec2 aTexCoord;
                out vec2 TexCoord;
                uniform mat4 transform;
                void main() {
                    gl_Position = transform * vec4(aPos, 0.0, 1.0);
                    TexCoord = aTexCoord;
                }
            """, GL_VERTEX_SHADER),
            shaders.compileShader("""
                #version 330 core
                out vec4 FragColor;
                in vec2 TexCoord;
                uniform sampler2D ourTexture;
                uniform float exposure;
                uniform float gamma;
                uniform vec3 wb;
                void main() {
                    vec4 texColor = texture(ourTexture, TexCoord);
                    vec3 color = texColor.rgb * wb * exposure;
                    color = clamp(color, 0.0, 1.0);
                    color = pow(color, vec3(1.0 / gamma));
                    FragColor = vec4(color, 1.0);
                }
            """, GL_FRAGMENT_SHADER)
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
        zoom_factor = 1.15
        if event.angleDelta().y() > 0: self.zoom *= zoom_factor
        else: self.zoom /= zoom_factor
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
        
    def set_custom_profile(self):
        file_name, _ = QFileDialog.getOpenFileName(self, "Select ICC Profile", "", "ICC Profiles (*.icc *.icm);;All Files (*)")
        if file_name:
            self.parent_editor.custom_profile_path = file_name
            self.lbl_custom_prof.setText(os.path.basename(file_name))

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
        self.linear_cache = None
        self.cache_dirty = True
        self.is_first_load = False
        
        self.counts_r = None
        self.counts_g = None
        self.counts_b = None
        
        # Color Management
        self.monitor_profile_bytes = get_x11_icc_profile()
        self.monitor_profile_path = get_colord_icc_file()
        self.custom_profile_path = None
        
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
        
        main_layout = QHBoxLayout(central_widget)
        
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
        control_layout = QVBoxLayout()
        main_layout.addLayout(control_layout, stretch=1)
        
        # Histogram
        self.histogram = HistogramWidget()
        control_layout.addWidget(self.histogram)
        
        btn_open = QPushButton("Open RAW Image")
        btn_open.clicked.connect(self.open_image)
        control_layout.addWidget(btn_open)
        
        # Fast preview checkbox
        self.chk_fast_preview = QCheckBox("Fast Preview (Half Size)")
        self.chk_fast_preview.setChecked(True)
        self.chk_fast_preview.stateChanged.connect(self.on_cache_invalidating_change)
        control_layout.addWidget(self.chk_fast_preview)
        
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
        
        # White Balance
        control_layout.addWidget(QLabel("White Balance:"))
        self.cmb_wb = QComboBox()
        self.cmb_wb.addItems(["Camera", "Auto"])
        self.cmb_wb.currentIndexChanged.connect(self.on_cache_invalidating_change)
        control_layout.addWidget(self.cmb_wb)
        
        # Temperature Offset Slider
        self.lbl_temp = ClickableLabel("Temp Offset: 0 K")
        self.lbl_temp.setToolTip("Click to reset")
        self.lbl_temp.clicked.connect(lambda: self.slider_temp.setValue(0))
        control_layout.addWidget(self.lbl_temp)
        self.slider_temp = QSlider(Qt.Orientation.Horizontal)
        self.slider_temp.setMinimum(-4000)
        self.slider_temp.setMaximum(4000)
        self.slider_temp.setValue(0)
        self.slider_temp.valueChanged.connect(self.on_wb_slider_changed)
        control_layout.addWidget(self.slider_temp)

        # Tint Offset Slider
        self.lbl_tint = ClickableLabel("Tint Offset: 0")
        self.lbl_tint.setToolTip("Click to reset")
        self.lbl_tint.clicked.connect(lambda: self.slider_tint.setValue(0))
        control_layout.addWidget(self.lbl_tint)
        self.slider_tint = QSlider(Qt.Orientation.Horizontal)
        self.slider_tint.setMinimum(-100)
        self.slider_tint.setMaximum(100)
        self.slider_tint.setValue(0)
        self.slider_tint.valueChanged.connect(self.on_wb_slider_changed)
        control_layout.addWidget(self.slider_tint)
        
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
        
        btn_settings = QPushButton("Settings")
        btn_settings.clicked.connect(self.open_settings)
        control_layout.addWidget(btn_settings)
        
        # Export
        control_layout.addStretch()
        btn_export = QPushButton("Export Image")
        btn_export.clicked.connect(self.export_image)
        control_layout.addWidget(btn_export)
        
        self.timer = QTimer()
        self.timer.setSingleShot(True)
        self.timer.timeout.connect(self.request_update_image)
        
    def open_image(self):
        file_name, _ = QFileDialog.getOpenFileName(self, "Open RAW Image", "", "RAW Files (*.cr2 *.CR2 *.cr3 *.CR3 *.nef *.NEF *.arw *.ARW *.dng *.DNG *.pef *.PEF);;All Files (*)")
        if file_name:
            self.raw_path = file_name
            try:
                if self.raw_image is not None:
                    self.raw_image.close()
                self.raw_image = rawpy.imread(self.raw_path)
                
                color_space = "sRGB"
                if os.path.basename(self.raw_path).startswith('_'):
                    color_space = "Adobe RGB"
                else:
                    try:
                        with open(self.raw_path, 'rb') as f:
                            tags = exifread.process_file(f, details=False)
                            if 'EXIF ColorSpace' in tags:
                                val = str(tags['EXIF ColorSpace'])
                                if 'Uncalibrated' in val or '65535' in val or 'Adobe' in val or '2' in val:
                                    color_space = "Adobe RGB"
                    except Exception:
                        pass
                
                self.cmb_colorspace.blockSignals(True)
                self.cmb_colorspace.setCurrentText(color_space)
                self.cmb_colorspace.blockSignals(False)
                
                self.cache_dirty = True
                self.is_first_load = True
                self.request_update_image()
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to load image: {str(e)}")
                
    def open_settings(self):
        old_mode = self.processing_mode
        old_profile = self.custom_profile_path
        
        dlg = SettingsDialog(self)
        if dlg.exec():
            self.processing_mode = dlg.cmb_mode.currentIndex()
            self.update_profile_label()
            
            if old_mode != self.processing_mode or old_profile != self.custom_profile_path:
                if self.processing_mode > 0 and self.linear_cache is None:
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

    def update_image_deferred(self):
        if self.raw_image is None:
            return
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
        if self.custom_profile_path:
            return ImageCms.ImageCmsProfile(self.custom_profile_path)
        if self.monitor_profile_bytes:
            return ImageCms.ImageCmsProfile(io.BytesIO(self.monitor_profile_bytes))
        if self.monitor_profile_path:
            return ImageCms.ImageCmsProfile(self.monitor_profile_path)
        return ImageCms.createProfile("sRGB")

    def process_and_display(self):
        ev = self.slider_exposure.value() / 6.0
        exposure_multiplier = 2.0 ** ev
        gamma_val = self.slider_gamma.value() / 100.0
        
        use_auto_wb = self.cmb_wb.currentText() == "Auto"
        use_camera_wb = self.cmb_wb.currentText() == "Camera"
        
        cs_text = self.cmb_colorspace.currentText()
        if cs_text == "sRGB":
            out_cs = rawpy.ColorSpace.sRGB
        elif cs_text == "Adobe RGB":
            out_cs = rawpy.ColorSpace.Adobe
        else:
            out_cs = rawpy.ColorSpace.ProPhoto
            
        half_size = self.chk_fast_preview.isChecked()
        
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
                
            self.gl_view.exposure = exposure_multiplier
            self.gl_view.gamma = gamma_val
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
            
            if self.processing_mode == 1:
                # Linear Cache Math
                arr = self.linear_cache.astype(np.float32) / 65535.0
                arr[:,:,0] *= off[0]
                arr[:,:,1] *= off[1]
                arr[:,:,2] *= off[2]
                arr = arr * exposure_multiplier
                arr = np.clip(arr, 0.0, 1.0)
                arr = np.power(arr, 1.0 / gamma_val) * 255.0
                self.processed_rgb = arr.astype(np.uint8)
            else:
                # LUT Optimization
                # For LUT, we can't easily bake WB offsets if they are per-channel
                # unless we have 3 LUTs. For now, we apply offsets to the final LUT indices.
                lut = np.arange(65536, dtype=np.float32) / 65535.0
                lut = lut * exposure_multiplier
                lut = np.clip(lut, 0.0, 1.0)
                lut = np.power(lut, 1.0 / gamma_val) * 255.0
                lut = lut.astype(np.uint8)
                
                res = lut[self.linear_cache]
                res = (res.astype(np.float32) * np.array(off)).clip(0, 255).astype(np.uint8)
                self.processed_rgb = res
        
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
            res = np.clip(res, 0.0, 1.0)
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
            elif cs_text == "Adobe RGB":
                out_cs = rawpy.ColorSpace.Adobe
            else:
                out_cs = rawpy.ColorSpace.ProPhoto
                
            export_rgb = self.raw_image.postprocess(
                use_camera_wb=use_camera_wb,
                use_auto_wb=use_auto_wb,
                half_size=False,
                exp_shift=exposure_multiplier,
                gamma=(gamma_val, 4.5),
                output_color=out_cs,
                output_bps=8
            )
            
            # Apply WB offsets
            off = self.calculate_wb_offsets()
            if off != [1.0, 1.0, 1.0]:
                export_f = export_rgb.astype(np.float32)
                export_f[:,:,0] *= off[0]
                export_f[:,:,1] *= off[1]
                export_f[:,:,2] *= off[2]
                export_rgb = np.clip(export_f, 0, 255).astype(np.uint8)
            
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

if __name__ == "__main__":
    app = QApplication(sys.argv)
    editor = RawEditor()
    editor.show()
    sys.exit(app.exec())
