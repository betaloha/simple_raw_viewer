import sys
import os
import subprocess
import io
import time
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QPushButton, QLabel, QFileDialog, 
                             QSlider, QComboBox, QCheckBox, QGraphicsView, 
                             QGraphicsScene, QGraphicsPixmapItem, QMessageBox,
                             QDialog, QFormLayout, QStackedWidget, QSpinBox,
                             QListWidget, QListWidgetItem, QGridLayout)
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
                uniform float blacks;
                uniform float shadows;
                uniform float highlights;
                uniform float whites;
                uniform float saturation;
                
                void main() {
                    vec4 texColor = texture(ourTexture, TexCoord);
                    vec3 color = texColor.rgb * wb * exposure;
                    
                    // Tonal Adjustments (Smooth overlapping bell curves)
                    for(int i=0; i<3; i++) {
                        float x = color[i];
                        
                        // Use smoothstep-based bell curves for natural transitions
                        float b_w = smoothstep(0.5, 0.0, x);
                        float s_w = smoothstep(0.0, 0.4, x) * smoothstep(0.8, 0.4, x);
                        float h_w = smoothstep(0.2, 0.6, x) * smoothstep(1.0, 0.6, x);
                        float w_w = smoothstep(0.5, 1.0, x);
                        
                        // Apply adjustments with smoother weighting
                        x += blacks * b_w * 0.05;
                        x *= (1.0 + shadows * s_w * 0.5);
                        x *= (1.0 + highlights * h_w * 0.5);
                        x += whites * w_w * 0.05;
                        
                        color[i] = clamp(x, 0.0, 1.0);
                    }
                    
                    // Saturation
                    float luma = dot(color, vec3(0.299, 0.587, 0.114));
                    color = mix(vec3(luma), color, saturation);
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
        
        glUniform1f(glGetUniformLocation(self.shader_program, "blacks"), getattr(self, 'blacks', 0.0))
        glUniform1f(glGetUniformLocation(self.shader_program, "shadows"), getattr(self, 'shadows', 0.0))
        glUniform1f(glGetUniformLocation(self.shader_program, "highlights"), getattr(self, 'highlights', 0.0))
        glUniform1f(glGetUniformLocation(self.shader_program, "whites"), getattr(self, 'whites', 0.0))
        glUniform1f(glGetUniformLocation(self.shader_program, "saturation"), 1.0 + getattr(self, 'saturation', 0.0))
        
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

from concurrent.futures import ProcessPoolExecutor, as_completed

class ThumbnailManagerSignals(QObject):
    result = pyqtSignal(int, bytes)

class ThumbnailManager(QRunnable):
    def __init__(self, file_paths):
        super().__init__()
        self.file_paths = file_paths
        self.signals = ThumbnailManagerSignals()

    def run(self):
        # We use a ProcessPoolExecutor to completely bypass the Python GIL
        with ProcessPoolExecutor() as executor:
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
        control_layout = QVBoxLayout()
        main_layout.addLayout(control_layout, stretch=1)
        
        # Histogram
        self.histogram = HistogramWidget()
        control_layout.addWidget(self.histogram)
        
        # Buttons Grid
        btn_grid = QGridLayout()
        control_layout.addLayout(btn_grid)
        
        btn_open = QPushButton("Open Folder")
        btn_open.clicked.connect(self.open_directory)
        btn_grid.addWidget(btn_open, 0, 0)

        btn_mimic = QPushButton("Auto-Match")
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
        tonal_grid = QFormLayout()
        control_layout.addLayout(tonal_grid)
        
        def create_tonal_slider(label_text, default_val, attr_name):
            lbl = ClickableLabel(f"{label_text}: 0")
            lbl.setToolTip("Click to reset")
            slider = QSlider(Qt.Orientation.Horizontal)
            slider.setRange(-100, 100)
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

        self.slider_blacks = create_tonal_slider("Blacks", 0, "blacks")
        self.slider_shadows = create_tonal_slider("Shadows", 0, "shadows")
        self.slider_highlights = create_tonal_slider("Highlights", 0, "highlights")
        self.slider_whites = create_tonal_slider("Whites", 0, "whites")
        self.slider_saturation = create_tonal_slider("Saturation", 0, "saturation")
        
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
        
        # Filmstrip
        self.filmstrip = QListWidget()
        self.filmstrip.setViewMode(QListWidget.ViewMode.IconMode)
        self.filmstrip.setIconSize(QSize(120, 80))
        self.filmstrip.setFlow(QListWidget.Flow.LeftToRight)
        self.filmstrip.setWrapping(False)
        self.filmstrip.setResizeMode(QListWidget.ResizeMode.Adjust)
        self.filmstrip.setFixedHeight(130)
        self.filmstrip.setHorizontalScrollMode(QListWidget.ScrollMode.ScrollPerPixel)
        self.filmstrip.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.filmstrip.itemClicked.connect(self.on_thumbnail_clicked)
        outer_layout.addWidget(self.filmstrip)
        
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
            
            # 7. Tone Curve Matching (Blacks, Shadows, Highlights, Whites)
            # Apply exposure and white balance to raw array to match target luminance space
            raw_adj = raw_arr * np.array([res_r * max(m_g, 1e-6), max(m_g, 1e-6), res_b * max(m_g, 1e-6)])
            
            # We match in Gamma space to optimize for the display Gamma
            luma_tgt = 0.299 * thumb_arr[:,:,0] + 0.587 * thumb_arr[:,:,1] + 0.114 * thumb_arr[:,:,2]
            luma_src = 0.299 * raw_adj[:,:,0] + 0.587 * raw_adj[:,:,1] + 0.114 * raw_adj[:,:,2]
            
            p_levels = [5, 25, 75, 95]
            tgt_p = np.percentile(luma_tgt, p_levels)
            src_p = np.percentile(luma_src, p_levels)
            
            def smoothstep(e0, e1, v):
                t = np.clip((v - e0) / (e1 - e0), 0.0, 1.0)
                return t * t * (3.0 - 2.0 * t)
                
            def eval_tonal(x, b, s, h, w, g):
                bw = smoothstep(0.5, 0.0, x)
                sw = smoothstep(0.0, 0.4, x) * smoothstep(0.8, 0.4, x)
                hw = smoothstep(0.2, 0.6, x) * smoothstep(1.0, 0.6, x)
                ww = smoothstep(0.5, 1.0, x)
                y = np.copy(x)
                y += b * bw * 0.05
                y *= (1.0 + s * sw * 0.5)
                y *= (1.0 + h * hw * 0.5)
                y += w * ww * 0.05
                y = np.clip(y, 1e-6, 1.0)
                return np.power(y, 1.0 / g)
                
            best_p = np.array([0.0, 0.0, 0.0, 0.0, 2.22]) # b, s, h, w, g
            lr = 5.0
            eps = 1e-4
            for _ in range(100):
                curr = eval_tonal(src_p, *best_p)
                err = curr - tgt_p
                
                grad = np.zeros(5)
                for i in range(5):
                    p_up = best_p.copy()
                    p_up[i] += eps
                    curr_up = eval_tonal(src_p, *p_up)
                    dcurr_dp = (curr_up - curr) / eps
                    grad[i] = np.sum(2 * err * dcurr_dp)
                    
                best_p -= lr * grad
                # Clip b, s, h, w to [-1, 1], and gamma to [1, 5]
                best_p[:4] = np.clip(best_p[:4], -1.0, 1.0)
                best_p[4] = np.clip(best_p[4], 1.0, 5.0)

            # Assign to sliders
            self.slider_blacks.setValue(int(best_p[0] * 100))
            self.slider_shadows.setValue(int(best_p[1] * 100))
            self.slider_highlights.setValue(int(best_p[2] * 100))
            self.slider_whites.setValue(int(best_p[3] * 100))
            self.slider_gamma.setValue(int(best_p[4] * 100))
            self.slider_saturation.setValue(int(np.clip(sat_val, -100, 100)))
            
            self.slider_temp.setValue(int(np.clip(best_dt, -4000, 4000)))
            self.slider_tint.setValue(int(np.clip(best_dtint, -100, 100)))
            
        except Exception as e:
            QMessageBox.warning(self, "Warning", f"Could not match thumbnail: {e}")
            
    def load_directory(self, dir_name):
        self.filmstrip.clear()
        self.image_files = []
        valid_extensions = ('.cr2', '.cr3', '.nef', '.arw', '.dng', '.pef')
        try:
            for f in sorted(os.listdir(dir_name)):
                if f.lower().endswith(valid_extensions):
                    full_path = os.path.join(dir_name, f)
                    self.image_files.append(full_path)
                    
                    item = QListWidgetItem(f)
                    self.filmstrip.addItem(item)
                    
            if self.image_files:
                # We no longer load the first image automatically to save time and prevent blocking.
                # The user will click an image in the filmstrip to start editing.
                pass
                
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
            item = self.filmstrip.item(index)
            item.setIcon(QIcon(QPixmap.fromImage(img)))

    def on_thumbnail_clicked(self, item):
        row = self.filmstrip.row(item)
        if 0 <= row < len(self.image_files):
            self.load_image(self.image_files[row])

    def load_image(self, file_path):
        self.raw_path = file_path
        try:
            if self.raw_image is not None:
                self.raw_image.close()
            self.raw_image = rawpy.imread(self.raw_path)
            
            color_space = self.get_default_colorspace(self.raw_path)
            
            self.cmb_colorspace.blockSignals(True)
            self.cmb_colorspace.setCurrentText(color_space)
            self.cmb_colorspace.blockSignals(False)
            
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
        
        self.slider_blacks.setValue(0)
        self.slider_shadows.setValue(0)
        self.slider_highlights.setValue(0)
        self.slider_whites.setValue(0)
        self.slider_saturation.setValue(0)
        
        self.slider_temp.setValue(0)
        self.slider_tint.setValue(0)
        self.on_wb_slider_changed() # Force labels and spin boxes update
        
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

    def apply_tonal_math(self, arr):
        # Localized tonal math for CPU paths
        blacks = self.blacks
        shadows = self.shadows
        highlights = self.highlights
        whites = self.whites
        
        res = arr.copy()
        
        # Helper for smoothstep-like behavior in numpy
        def n_smoothstep(edge0, edge1, x):
            t = np.clip((x - edge0) / (edge1 - edge0), 0.0, 1.0)
            return t * t * (3.0 - 2.0 * t)

        b_w = n_smoothstep(0.5, 0.0, res)
        s_w = n_smoothstep(0.0, 0.4, res) * n_smoothstep(0.8, 0.4, res)
        h_w = n_smoothstep(0.2, 0.6, res) * n_smoothstep(1.0, 0.6, res)
        w_w = n_smoothstep(0.5, 1.0, res)
        
        res += blacks * b_w * 0.05
        res *= (1.0 + shadows * s_w * 0.5)
        res *= (1.0 + highlights * h_w * 0.5)
        res += whites * w_w * 0.05
        
        return np.clip(res, 0.0, 1.0)

    def apply_saturation(self, arr, sat_mult):
        if sat_mult == 1.0:
            return arr
            
        was_uint8 = False
        if arr.dtype == np.uint8:
            arr = arr.astype(np.float32) / 255.0
            was_uint8 = True
            
        luma = 0.299 * arr[:,:,0] + 0.587 * arr[:,:,1] + 0.114 * arr[:,:,2]
        res = np.empty_like(arr)
        res[:,:,0] = luma + (arr[:,:,0] - luma) * sat_mult
        res[:,:,1] = luma + (arr[:,:,1] - luma) * sat_mult
        res[:,:,2] = luma + (arr[:,:,2] - luma) * sat_mult
        
        if was_uint8:
            return np.clip(res * 255.0, 0, 255).astype(np.uint8)
        return np.clip(res, 0.0, 1.0)

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
                
            self.gl_view.exposure = exposure_multiplier
            self.gl_view.gamma = gamma_val
            self.gl_view.blacks = self.blacks
            self.gl_view.shadows = self.shadows
            self.gl_view.highlights = self.highlights
            self.gl_view.whites = self.whites
            self.gl_view.saturation = getattr(self, 'saturation', 0.0)
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
                arr = self.apply_tonal_math(arr) # <---
                arr = np.power(arr, 1.0 / gamma_val) * 255.0
                self.processed_rgb = arr.astype(np.uint8)
            else:
                # LUT Optimization
                lut = np.arange(65536, dtype=np.float32) / 65535.0
                lut = lut * exposure_multiplier
                lut = self.apply_tonal_math(lut) # <---
                lut = np.power(lut, 1.0 / gamma_val) * 255.0
                lut = lut.astype(np.uint8)
                
                res = lut[self.linear_cache]
                res = (res.astype(np.float32) * np.array(off)).clip(0, 255).astype(np.uint8)
                self.processed_rgb = res
                
        # Saturation (CPU)
        sat_mult = 1.0 + getattr(self, 'saturation', 0.0)
        if sat_mult != 1.0:
            self.processed_rgb = self.apply_saturation(self.processed_rgb, sat_mult)
        
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
            elif cs_text == "Adobe RGB":
                out_cs = rawpy.ColorSpace.Adobe
            else:
                out_cs = rawpy.ColorSpace.ProPhoto
                
            # Process to 16-bit linear for high-quality math
            export_lin = self.raw_image.postprocess(
                use_camera_wb=use_camera_wb,
                use_auto_wb=use_auto_wb,
                half_size=False,
                exp_shift=exposure_multiplier,
                gamma=(1, 1),
                output_color=out_cs,
                output_bps=16,
                no_auto_bright=True
            )
            
            # Apply WB offsets and Tonal adjustments in 16-bit space
            off = self.calculate_wb_offsets()
            arr = export_lin.astype(np.float32) / 65535.0
            arr[:,:,0] *= off[0]
            arr[:,:,1] *= off[1]
            arr[:,:,2] *= off[2]
            
            arr = self.apply_tonal_math(arr)
            
            sat_mult = 1.0 + getattr(self, 'saturation', 0.0)
            if sat_mult != 1.0:
                arr = self.apply_saturation(arr, sat_mult)
            
            # Apply Gamma
            arr = np.power(arr, 1.0 / gamma_val) * 255.0
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

if __name__ == "__main__":
    app = QApplication(sys.argv)
    editor = RawEditor()
    editor.show()
    sys.exit(app.exec())
