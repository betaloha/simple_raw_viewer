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
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QPixmap, QImage
import rawpy
import numpy as np
from PIL import Image, ImageCms
from OpenGL.GL import *
from OpenGL.GL import shaders
import exifread

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
                void main() {
                    vec4 texColor = texture(ourTexture, TexCoord);
                    vec3 color = texColor.rgb * exposure;
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
        
        btn_open = QPushButton("Open RAW Image")
        btn_open.clicked.connect(self.open_image)
        control_layout.addWidget(btn_open)
        
        # Fast preview checkbox
        self.chk_fast_preview = QCheckBox("Fast Preview (Half Size)")
        self.chk_fast_preview.setChecked(True)
        self.chk_fast_preview.stateChanged.connect(self.on_cache_invalidating_change)
        control_layout.addWidget(self.chk_fast_preview)
        
        # Exposure Slider
        self.lbl_exposure = QLabel("Exposure: 0.00 EV")
        control_layout.addWidget(self.lbl_exposure)
        self.slider_exposure = QSlider(Qt.Orientation.Horizontal)
        self.slider_exposure.setMinimum(-30) # -5.0 stops
        self.slider_exposure.setMaximum(30)  # +5.0 stops
        self.slider_exposure.setValue(0)     # 0 stops
        self.slider_exposure.valueChanged.connect(self.on_exposure_changed)
        control_layout.addWidget(self.slider_exposure)
        
        # Gamma Slider
        self.lbl_gamma = QLabel("Gamma: 2.22")
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
            
            if self.cache_dirty or self.linear_cache is None:
                QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
                self.linear_cache = self.raw_image.postprocess(
                    use_camera_wb=use_camera_wb,
                    use_auto_wb=use_auto_wb,
                    half_size=half_size,
                    output_color=out_cs,
                    output_bps=16,
                    gamma=(1, 1),
                    no_auto_bright=False
                )
                self.cache_dirty = False
                h, w, _ = self.linear_cache.shape
                self.gl_view.set_image(self.linear_cache, w, h)
                QApplication.restoreOverrideCursor()
                
            if self.is_first_load:
                self.gl_view.fit_in_view()
                self.is_first_load = False
                
            self.gl_view.exposure = exposure_multiplier
            self.gl_view.gamma = gamma_val
            self.gl_view.update()
            
            self.target_icc_path = self.system_icc_paths.get(cs_text)
            return

        # === CPU Modes ===
        self.stack.setCurrentWidget(self.view)
        
        if self.processing_mode == 0:
            # Full rawpy processing (Slow)
            self.processed_rgb = self.raw_image.postprocess(
                use_camera_wb=use_camera_wb,
                use_auto_wb=use_auto_wb,
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
                    use_camera_wb=use_camera_wb,
                    use_auto_wb=use_auto_wb,
                    half_size=half_size,
                    output_color=out_cs,
                    output_bps=16,
                    gamma=(1, 1),
                    no_auto_bright=False
                )
                self.cache_dirty = False
                
            if self.processing_mode == 1:
                # Linear Cache Math
                arr = self.linear_cache.astype(np.float32) / 65535.0
                arr = arr * exposure_multiplier
                arr = np.clip(arr, 0.0, 1.0)
                arr = np.power(arr, 1.0 / gamma_val) * 255.0
                self.processed_rgb = arr.astype(np.uint8)
            else:
                # LUT Optimization
                lut = np.arange(65536, dtype=np.float32) / 65535.0
                lut = lut * exposure_multiplier
                lut = np.clip(lut, 0.0, 1.0)
                lut = np.power(lut, 1.0 / gamma_val) * 255.0
                lut = lut.astype(np.uint8)
                self.processed_rgb = lut[self.linear_cache]
        
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
