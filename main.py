import sys
import os
import subprocess
import io
import time
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QPushButton, QLabel, QFileDialog, 
                             QSlider, QComboBox, QCheckBox, QGraphicsView, 
                             QGraphicsScene, QGraphicsPixmapItem, QMessageBox)
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QPixmap, QImage
import rawpy
import numpy as np
from PIL import Image, ImageCms

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

class RawEditor(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("RAW Image Editor")
        self.resize(1200, 800)
        
        self.raw_path = None
        self.raw_image = None
        self.processed_rgb = None
        self.target_icc_path = None
        
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
        
        # Left Panel - Image View
        self.view = QGraphicsView()
        self.scene = QGraphicsScene()
        self.view.setScene(self.scene)
        self.pixmap_item = QGraphicsPixmapItem()
        self.scene.addItem(self.pixmap_item)
        main_layout.addWidget(self.view, stretch=3)
        
        # Right Panel - Controls
        control_layout = QVBoxLayout()
        main_layout.addLayout(control_layout, stretch=1)
        
        btn_open = QPushButton("Open RAW Image")
        btn_open.clicked.connect(self.open_image)
        control_layout.addWidget(btn_open)
        
        # Fast preview checkbox
        self.chk_fast_preview = QCheckBox("Fast Preview (Half Size)")
        self.chk_fast_preview.setChecked(True)
        self.chk_fast_preview.stateChanged.connect(self.request_update_image)
        control_layout.addWidget(self.chk_fast_preview)
        
        # Brightness Slider
        control_layout.addWidget(QLabel("Brightness (Exposure):"))
        self.slider_brightness = QSlider(Qt.Orientation.Horizontal)
        self.slider_brightness.setMinimum(10) # 0.1
        self.slider_brightness.setMaximum(500) # 5.0
        self.slider_brightness.setValue(100) # 1.0
        self.slider_brightness.valueChanged.connect(self.update_image_deferred)
        control_layout.addWidget(self.slider_brightness)
        
        # White Balance
        control_layout.addWidget(QLabel("White Balance:"))
        self.cmb_wb = QComboBox()
        self.cmb_wb.addItems(["Camera", "Auto"])
        self.cmb_wb.currentIndexChanged.connect(self.request_update_image)
        control_layout.addWidget(self.cmb_wb)
        
        # Output Color Space
        control_layout.addWidget(QLabel("Output Color Space:"))
        self.cmb_colorspace = QComboBox()
        self.cmb_colorspace.addItems(["sRGB", "Adobe RGB", "ProPhoto RGB"])
        self.cmb_colorspace.currentIndexChanged.connect(self.request_update_image)
        control_layout.addWidget(self.cmb_colorspace)
        
        # Monitor ICC Profile
        if self.monitor_profile_bytes:
            prof_text = "Auto-detected (X11 _ICC_PROFILE)"
        elif self.monitor_profile_path:
            prof_text = f"Auto-detected (colord)\n{os.path.basename(self.monitor_profile_path)}"
        else:
            prof_text = "None detected (Fallback to sRGB)"
            
        self.lbl_profile = QLabel(f"Monitor Profile:\n{prof_text}")
        self.lbl_profile.setWordWrap(True)
        control_layout.addWidget(self.lbl_profile)
        
        btn_custom_profile = QPushButton("Set Custom Display Profile")
        btn_custom_profile.clicked.connect(self.set_custom_profile)
        control_layout.addWidget(btn_custom_profile)
        
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
                self.request_update_image()
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to load image: {str(e)}")
                
    def set_custom_profile(self):
        file_name, _ = QFileDialog.getOpenFileName(self, "Select ICC Profile", "", "ICC Profiles (*.icc *.icm);;All Files (*)")
        if file_name:
            self.custom_profile_path = file_name
            self.lbl_profile.setText(f"Monitor Profile:\nCustom: {os.path.basename(file_name)}")
            self.request_update_image()

    def update_image_deferred(self):
        if self.raw_image is None:
            return
        self.timer.start(100) # 100ms debounce

    def request_update_image(self):
        if self.raw_image is None:
            return
        
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            self.process_and_display()
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to process image: {str(e)}")
        finally:
            QApplication.restoreOverrideCursor()

    def get_display_profile(self):
        if self.custom_profile_path:
            return ImageCms.ImageCmsProfile(self.custom_profile_path)
        if self.monitor_profile_bytes:
            return ImageCms.ImageCmsProfile(io.BytesIO(self.monitor_profile_bytes))
        if self.monitor_profile_path:
            return ImageCms.ImageCmsProfile(self.monitor_profile_path)
        # Fallback to sRGB if no display profile
        return ImageCms.createProfile("sRGB")

    def process_and_display(self):
        brightness = self.slider_brightness.value() / 100.0
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
        
        # rawpy processing
        self.processed_rgb = self.raw_image.postprocess(
            use_camera_wb=use_camera_wb,
            use_auto_wb=use_auto_wb,
            half_size=half_size,
            exp_shift=brightness,
            output_color=out_cs,
            output_bps=8
        )
        
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
        
        # Fit in view
        self.view.fitInView(self.scene.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)

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
            brightness = self.slider_brightness.value() / 100.0
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
                exp_shift=brightness,
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
            else:
                img.save(file_name)
                
            QMessageBox.information(self, "Success", f"Image successfully exported to:\n{file_name}")
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to export image: {str(e)}")
        finally:
            QApplication.restoreOverrideCursor()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    editor = RawEditor()
    editor.show()
    sys.exit(app.exec())
