import sys
import numpy as np
from PyQt5.QtWidgets import (QApplication, QMainWindow, QVBoxLayout, QHBoxLayout, 
                             QLabel, QPushButton, QSlider, QFileDialog, QWidget, 
                             QGroupBox, QMessageBox, QCheckBox, QTextEdit, QTabWidget)
from PyQt5.QtGui import QImage, QPixmap, QPainter, QColor, QFont
from PyQt5.QtCore import Qt, QTimer, QPoint, QRect
import cv2
import math
import time
import mediapipe as mp
import os

class SliceViewConverter(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("SliceView Video Converter")
        self.setGeometry(100, 100, 1400, 900)
        
        # Video processing attributes
        self.video_capture = None
        self.current_frame = None
        self.is_playing = False
        self.frame_rate = 30  # Default FPS
        self.frame_count = 0
        self.processing_time = 0
        
        # Layer data
        self.num_layers = 3
        self.layers = []
        self.layer_pixmaps = []
        
        # Person segmentation attributes
        self.mp_selfie_segmentation = mp.solutions.selfie_segmentation
        self.selfie_segmentation = self.mp_selfie_segmentation.SelfieSegmentation(model_selection=1)
        self.person_mask = None
        
        # Performance metrics
        self.last_frame_time = 0
        self.fps_counter = 0
        self.fps = 0
        
        # Create central widget with tabs
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        main_layout = QVBoxLayout()
        self.central_widget.setLayout(main_layout)
        
        # Create tab widget
        self.tab_widget = QTabWidget()
        main_layout.addWidget(self.tab_widget)
        
        # Create tabs
        self.create_preview_tab()
        self.create_slices_tab()
        self.create_combined_view_tab()
        
        # Create control area
        control_area = QWidget()
        control_layout = QVBoxLayout()
        control_area.setLayout(control_layout)
        
        # Top controls (file handling and playback)
        top_controls = QHBoxLayout()
        
        # Load Video Button
        self.load_button = QPushButton("Load Video")
        self.load_button.clicked.connect(self.load_video)
        top_controls.addWidget(self.load_button)
        
        # Play/Pause Button
        self.play_button = QPushButton("Play")
        self.play_button.clicked.connect(self.toggle_playback)
        top_controls.addWidget(self.play_button)
        
        # Export Button
        self.export_button = QPushButton("Export Sliced Video")
        self.export_button.clicked.connect(self.export_sliced_video)
        top_controls.addWidget(self.export_button)
        
        control_layout.addLayout(top_controls)
        
        # Middle controls (settings)
        middle_controls = QHBoxLayout()
        
        # Layer settings
        layer_group = QGroupBox("Layer Settings")
        layer_layout = QVBoxLayout()
        
        # Number of layers slider
        layer_count_layout = QHBoxLayout()
        layer_count_layout.addWidget(QLabel("Number of Layers:"))
        self.layer_count_slider = QSlider(Qt.Horizontal)
        self.layer_count_slider.setRange(2, 5)
        self.layer_count_slider.setValue(self.num_layers)
        self.layer_count_slider.valueChanged.connect(self.update_layer_count)
        layer_count_layout.addWidget(self.layer_count_slider)
        self.layer_count_label = QLabel(f"{self.num_layers}")
        layer_count_layout.addWidget(self.layer_count_label)
        layer_layout.addLayout(layer_count_layout)
        
        # Layer opacity
        opacity_layout = QHBoxLayout()
        opacity_layout.addWidget(QLabel("Layer Opacity:"))
        self.opacity_slider = QSlider(Qt.Horizontal)
        self.opacity_slider.setRange(1, 100)
        self.opacity_slider.setValue(80)
        self.opacity_slider.valueChanged.connect(self.update_settings)
        opacity_layout.addWidget(self.opacity_slider)
        layer_layout.addLayout(opacity_layout)
        
        layer_group.setLayout(layer_layout)
        middle_controls.addWidget(layer_group)
        
        # Segmentation settings
        seg_group = QGroupBox("Segmentation Settings")
        seg_layout = QVBoxLayout()
        
        # Person segmentation toggle
        person_seg_layout = QHBoxLayout()
        self.person_seg_checkbox = QCheckBox("Person Segmentation")
        self.person_seg_checkbox.setChecked(True)
        self.person_seg_checkbox.stateChanged.connect(self.update_settings)
        person_seg_layout.addWidget(self.person_seg_checkbox)
        seg_layout.addLayout(person_seg_layout)
        
        # Segmentation threshold
        threshold_layout = QHBoxLayout()
        threshold_layout.addWidget(QLabel("Segmentation Threshold:"))
        self.threshold_slider = QSlider(Qt.Horizontal)
        self.threshold_slider.setRange(1, 95)
        self.threshold_slider.setValue(30)
        self.threshold_slider.valueChanged.connect(self.update_settings)
        threshold_layout.addWidget(self.threshold_slider)
        seg_layout.addLayout(threshold_layout)
        
        # Edge enhancement toggle
        edge_layout = QHBoxLayout()
        self.edge_checkbox = QCheckBox("Edge Enhancement")
        self.edge_checkbox.setChecked(True)
        self.edge_checkbox.stateChanged.connect(self.update_settings)
        edge_layout.addWidget(self.edge_checkbox)
        seg_layout.addLayout(edge_layout)
        
        seg_group.setLayout(seg_layout)
        middle_controls.addWidget(seg_group)
        
        # Resolution settings
        res_group = QGroupBox("Resolution")
        res_layout = QVBoxLayout()
        
        resolution_layout = QHBoxLayout()
        resolution_layout.addWidget(QLabel("Processing Resolution:"))
        self.resolution_slider = QSlider(Qt.Horizontal)
        self.resolution_slider.setRange(64, 512)
        self.resolution_slider.setValue(256)
        self.resolution_slider.valueChanged.connect(self.update_settings)
        resolution_layout.addWidget(self.resolution_slider)
        self.resolution_label = QLabel("256")
        resolution_layout.addWidget(self.resolution_label)
        res_layout.addLayout(resolution_layout)
        
        res_group.setLayout(res_layout)
        middle_controls.addWidget(res_group)
        
        control_layout.addLayout(middle_controls)
        
        # Add debug output area
        debug_group = QGroupBox("Debug Output")
        debug_layout = QVBoxLayout()
        
        self.debug_output = QTextEdit()
        self.debug_output.setReadOnly(True)
        self.debug_output.setMaximumHeight(100)
        debug_layout.addWidget(self.debug_output)
        
        debug_group.setLayout(debug_layout)
        control_layout.addWidget(debug_group)
        
        # Status bar
        self.status_label = QLabel("Status: Ready")
        control_layout.addWidget(self.status_label)
        
        main_layout.addWidget(control_area)
        
        # Create timers for video playback and FPS calculation
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        
        self.fps_timer = QTimer()
        self.fps_timer.timeout.connect(self.calculate_fps)
        self.fps_timer.start(1000)  # Update every second
        
        # Initialize layers
        self.init_layers()
        
        self.log_debug("SliceView Converter initialized")

    def create_preview_tab(self):
        """Create the original video preview tab"""
        preview_tab = QWidget()
        layout = QVBoxLayout()
        
        self.preview_label = QLabel("No video loaded")
        self.preview_label.setAlignment(Qt.AlignCenter)
        self.preview_label.setMinimumSize(800, 600)
        self.preview_label.setStyleSheet("background-color: black;")
        
        layout.addWidget(self.preview_label)
        preview_tab.setLayout(layout)
        
        self.tab_widget.addTab(preview_tab, "Original Video")

    def create_slices_tab(self):
        """Create tab showing individual layers"""
        slices_tab = QWidget()
        layout = QVBoxLayout()
        
        self.layers_widget = QWidget()
        self.layers_layout = QHBoxLayout()
        self.layers_widget.setLayout(self.layers_layout)
        
        layout.addWidget(self.layers_widget)
        slices_tab.setLayout(layout)
        
        self.tab_widget.addTab(slices_tab, "Layer Slices")

    def create_combined_view_tab(self):
        """Create tab showing combined sliceview representation"""
        combined_tab = QWidget()
        layout = QVBoxLayout()
        
        self.combined_label = QLabel("No video loaded")
        self.combined_label.setAlignment(Qt.AlignCenter)
        self.combined_label.setMinimumSize(800, 600)
        self.combined_label.setStyleSheet("background-color: black;")
        
        layout.addWidget(self.combined_label)
        combined_tab.setLayout(layout)
        
        self.tab_widget.addTab(combined_tab, "SliceView Preview")

    def init_layers(self):
        """Initialize layer display widgets"""
        # Clear any existing layers
        for i in reversed(range(self.layers_layout.count())): 
            self.layers_layout.itemAt(i).widget().setParent(None)
        
        # Create new layer widgets
        self.layer_labels = []
        for i in range(self.num_layers):
            label = QLabel(f"Layer {i+1}")
            label.setAlignment(Qt.AlignCenter)
            label.setMinimumSize(300, 300)
            label.setStyleSheet("background-color: black;")
            self.layer_labels.append(label)
            self.layers_layout.addWidget(label)
        
        # Initialize layer data
        self.layers = [None] * self.num_layers
        self.layer_pixmaps = [None] * self.num_layers

    def update_layer_count(self):
        """Update the number of layers based on slider value"""
        self.num_layers = self.layer_count_slider.value()
        self.layer_count_label.setText(f"{self.num_layers}")
        self.init_layers()
        self.log_debug(f"Layer count updated to {self.num_layers}")
        
        # Reprocess current frame if available
        if self.current_frame is not None:
            self.process_frame(self.current_frame)

    def update_settings(self):
        """Update processing settings based on UI controls"""
        # Update resolution label
        self.resolution_label.setText(f"{self.resolution_slider.value()}")
        
        # Reprocess current frame if available
        if self.current_frame is not None:
            self.process_frame(self.current_frame)

    def log_debug(self, message):
        """Add debug message to the debug output area"""
        timestamp = time.strftime("%H:%M:%S")
        self.debug_output.append(f"[{timestamp}] {message}")
        
        # Auto-scroll to the bottom
        scrollbar = self.debug_output.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())

    def calculate_fps(self):
        """Calculate and display current frames per second"""
        self.fps = self.fps_counter
        self.fps_counter = 0
        if self.is_playing:
            self.log_debug(f"FPS: {self.fps}, Processing time: {self.processing_time:.2f}ms per frame")

    def load_video(self):
        """Load a video file for processing"""
        try:
            # Open file dialog
            file_path, _ = QFileDialog.getOpenFileName(
                self, "Open Video", "", "Video Files (*.mp4 *.avi *.mov *.mkv)"
            )
            
            if not file_path:
                return
            
            # Update status
            self.status_label.setText(f"Loading video: {file_path}")
            self.log_debug(f"Loading video file: {file_path}")
            
            # Close any existing video
            if self.video_capture is not None:
                self.video_capture.release()
            
            # Open video file
            self.video_capture = cv2.VideoCapture(file_path)
            
            if not self.video_capture.isOpened():
                raise Exception("Failed to open video file")
            
            # Get video properties
            self.frame_rate = self.video_capture.get(cv2.CAP_PROP_FPS)
            frame_count = int(self.video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
            frame_width = int(self.video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
            frame_height = int(self.video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            self.log_debug(f"Video properties: {frame_width}x{frame_height}, {self.frame_rate} FPS, {frame_count} frames")
            
            if self.frame_rate <= 0:
                self.frame_rate = 30
            
            # Read first frame
            success, frame = self.video_capture.read()
            if not success:
                raise Exception("Failed to read video frame")
            
            # Process first frame
            self.current_frame = frame
            self.process_frame(frame)
            
            # Update status
            self.status_label.setText("Video loaded successfully")
            self.play_button.setText("Play")
            self.is_playing = False
            
        except Exception as e:
            self.log_debug(f"ERROR: {str(e)}")
            QMessageBox.critical(self, "Error", f"Failed to load video: {str(e)}")
            self.status_label.setText("Error loading video")

    def toggle_playback(self):
        """Toggle video playback"""
        if not self.video_capture:
            QMessageBox.warning(self, "Warning", "No video loaded")
            return
        
        if self.is_playing:
            # Pause
            self.timer.stop()
            self.play_button.setText("Play")
            self.status_label.setText("Playback paused")
        else:
            # Play
            interval = int(1000 / self.frame_rate)
            self.timer.start(interval)
            self.play_button.setText("Pause")
            self.status_label.setText("Playing video")
        
        self.is_playing = not self.is_playing

    def update_frame(self):
        """Read and process the next video frame"""
        if not self.video_capture:
            return
        
        # Measure frame processing time
        start_time = time.time()
        
        # Read next frame
        success, frame = self.video_capture.read()
        
        if not success:
            # End of video
            self.timer.stop()
            self.video_capture.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Rewind
            self.play_button.setText("Play")
            self.is_playing = False
            self.status_label.setText("End of video")
            return
        
        # Process frame
        self.current_frame = frame
        self.frame_count += 1
        self.process_frame(frame)
        
        # Calculate processing time
        end_time = time.time()
        self.processing_time = (end_time - start_time) * 1000
        self.fps_counter += 1

    def segment_person(self, frame):
        """Segment person from the background using MediaPipe"""
        if not self.person_seg_checkbox.isChecked():
            return None
        
        try:
            # Convert to RGB for MediaPipe
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Process the frame with MediaPipe
            results = self.selfie_segmentation.process(rgb_frame)
            
            # Get segmentation mask
            if results.segmentation_mask is not None:
                # Convert to binary mask with threshold
                threshold = self.threshold_slider.value() / 100.0
                mask = (results.segmentation_mask > threshold).astype(np.uint8) * 255
                
                # Apply morphological operations to clean the mask
                kernel = np.ones((5, 5), np.uint8)
                mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
                mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
                
                return mask
            
            return None
        except Exception as e:
            self.log_debug(f"ERROR: {str(e)}")
            return None

    def process_frame(self, frame):
        """Process a video frame into multiple depth layers"""
        try:
            # Get current resolution setting
            resolution = self.resolution_slider.value()
            
            # Calculate aspect ratio for resizing
            h, w = frame.shape[:2]
            aspect = w / h
            
            # Resize frame (maintain aspect ratio)
            if w > h:
                new_w = resolution
                new_h = int(resolution / aspect)
            else:
                new_h = resolution
                new_w = int(resolution * aspect)
            
            resized = cv2.resize(frame, (new_w, new_h))
            
            # Display original frame
            self.display_original(resized)
            
            # Segment person if enabled
            person_mask = self.segment_person(resized)
            
            # Create depth map
            gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
            depth_map = cv2.GaussianBlur(gray, (3, 3), 0)
            
            # Apply edge enhancement if enabled
            if self.edge_checkbox.isChecked():
                edges = cv2.Canny(gray, 100, 200)
                edges = cv2.dilate(edges, None)
                depth_map = cv2.addWeighted(depth_map, 0.7, edges, 0.3, 0)
            
            # Create layer masks based on person segmentation and depth map
            if person_mask is not None:
                # Create distance transform for background (distance from person)
                bg_mask = 255 - person_mask
                dist_transform = cv2.distanceTransform(bg_mask, cv2.DIST_L2, 5)
                dist_transform = cv2.normalize(dist_transform, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
                
                # Create layer masks with person in foreground, background at varying depths
                layer_masks = self.create_masks_with_person(person_mask, dist_transform, depth_map)
            else:
                # Create layer masks based only on depth
                layer_masks = self.create_masks_from_depth(depth_map)
            
            # Create layer images
            opacity = self.opacity_slider.value() / 100.0
            self.create_layer_images(resized, layer_masks, opacity)
            
            # Update layer displays
            self.update_layer_displays()
            
            # Create combined sliceview representation
            self.create_combined_view()
            
        except Exception as e:
            self.log_debug(f"ERROR processing frame: {str(e)}")

    def create_masks_with_person(self, person_mask, dist_transform, depth_map):
        """Create layer masks with person in foreground and background at varying depths"""
        h, w = depth_map.shape
        masks = []
        
        # Person goes in the foreground (first) layer
        foreground_mask = person_mask.copy()
        masks.append(foreground_mask)
        
        # Create masks for remaining layers based on distance from person
        if self.num_layers > 1:
            # Calculate depth ranges for background layers
            ranges = np.linspace(0, 255, self.num_layers)
            
            for i in range(1, self.num_layers):
                min_val = ranges[i-1]
                max_val = ranges[i]
                
                # Create mask for this depth range
                layer_mask = np.zeros((h, w), dtype=np.uint8)
                mask_condition = (dist_transform >= min_val) & (dist_transform < max_val) & (person_mask == 0)
                layer_mask[mask_condition] = 255
                
                masks.append(layer_mask)
        
        return masks

    def create_masks_from_depth(self, depth_map):
        """Create layer masks based only on depth information"""
        h, w = depth_map.shape
        masks = []
        
        # Calculate depth ranges for layers
        depth_min = np.min(depth_map)
        depth_max = np.max(depth_map)
        step = (depth_max - depth_min) / self.num_layers
        
        for i in range(self.num_layers):
            lower = depth_min + i * step
            upper = depth_min + (i + 1) * step
            
            # Create mask for this depth range
            mask = np.zeros((h, w), dtype=np.uint8)
            mask[(depth_map >= lower) & (depth_map < upper)] = 255
            
            masks.append(mask)
        
        return masks

    def create_layer_images(self, image, masks, opacity):
        """Create RGBA images for each layer"""
        h, w = image.shape[:2]
        self.layers = []
        
        for i, mask in enumerate(masks):
            # Create transparent layer image
            layer = np.zeros((h, w, 4), dtype=np.uint8)
            
            # Copy RGB channels from original image
            layer[..., :3] = image
            
            # Set alpha channel from mask (with opacity)
            layer[..., 3] = cv2.multiply(mask, int(opacity * 255))
            
            self.layers.append(layer)

    def update_layer_displays(self):
        """Update the layer preview displays"""
        for i, layer in enumerate(self.layers):
            if i < len(self.layer_labels):
                # Convert layer to QPixmap
                h, w = layer.shape[:2]
                qimg = QImage(layer.data, w, h, layer.strides[0], QImage.Format_RGBA8888)
                pixmap = QPixmap.fromImage(qimg)
                
                # Store pixmap for combined view
                self.layer_pixmaps[i] = pixmap
                
                # Display in layer preview
                self.layer_labels[i].setPixmap(pixmap.scaled(
                    self.layer_labels[i].width(),
                    self.layer_labels[i].height(),
                    Qt.KeepAspectRatio,
                    Qt.SmoothTransformation
                ))

    def display_original(self, frame):
        """Display the original frame"""
        # Convert to QPixmap
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w = rgb_frame.shape[:2]
        qimg = QImage(rgb_frame.data, w, h, rgb_frame.strides[0], QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qimg)
        
        # Display in preview
        self.preview_label.setPixmap(pixmap.scaled(
            self.preview_label.width(),
            self.preview_label.height(),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation
        ))

    def create_combined_view(self):
        """Create combined SliceView representation"""
        if not self.layer_pixmaps or any(pixmap is None for pixmap in self.layer_pixmaps):
            return
        
        # Create combined image (side by side)
        combined_width = sum(pixmap.width() for pixmap in self.layer_pixmaps)
        max_height = max(pixmap.height() for pixmap in self.layer_pixmaps)
        
        combined = QPixmap(combined_width, max_height)
        combined.fill(Qt.black)
        
        # Draw layers
        painter = QPainter(combined)
        x_pos = 0
        for pixmap in self.layer_pixmaps:
            # Draw with vertical mirroring (as required by SliceView)
            painter.drawPixmap(x_pos, 0, pixmap)
            x_pos += pixmap.width()
        
        painter.end()
        
        # Draw layer separators and labels
        painter = QPainter(combined)
        painter.setPen(QColor(255, 255, 255))
        painter.setFont(QFont("Arial", 12))
        
        x_pos = 0
        for i, pixmap in enumerate(self.layer_pixmaps):
            # Draw separator line
            if i > 0:
                painter.drawLine(x_pos, 0, x_pos, max_height)
            
            # Draw layer label
            painter.drawText(
                QRect(x_pos + 5, 5, pixmap.width() - 10, 30),
                Qt.AlignLeft | Qt.AlignTop,
                f"Layer {i+1}"
            )
            
            x_pos += pixmap.width()
        
        painter.end()
        
        # Display in combined view
        self.combined_label.setPixmap(combined.scaled(
            self.combined_label.width(),
            self.combined_label.height(),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation
        ))

    def export_sliced_video(self):
        """Export the sliced video for SliceView display"""
        if not self.video_capture:
            QMessageBox.warning(self, "Warning", "No video loaded")
            return
        
        try:
            # Get output file name
            file_path, _ = QFileDialog.getSaveFileName(
                self, "Save SliceView Video", "", "Video Files (*.mp4 *.avi)"
            )
            
            if not file_path:
                return
            
            # Rewind video to start
            self.video_capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
            
            # Get video properties
            fps = self.video_capture.get(cv2.CAP_PROP_FPS)
            if fps <= 0:
                fps = 30
            
            # Create video writer
            # First determine the output size from a sample frame
            success, frame = self.video_capture.read()
            if not success:
                raise Exception("Failed to read video frame")
            
            # Process frame to get layer images
            self.process_frame(frame)
            
            if not self.layer_pixmaps or any(pixmap is None for pixmap in self.layer_pixmaps):
                raise Exception("Failed to create layer images")
            
            # Calculate output dimensions
            combined_width = sum(pixmap.width() for pixmap in self.layer_pixmaps)
            max_height = max(pixmap.height() for pixmap in self.layer_pixmaps)
            
            # Create video writer
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            writer = cv2.VideoWriter(file_path, fourcc, fps, (combined_width, max_height))
            
            # Rewind video again
            self.video_capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
            
            # Process all frames
            frame_count = int(self.video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
            progress_dialog = QMessageBox()
            progress_dialog.setIcon(QMessageBox.Information)
            progress_dialog.setWindowTitle("Exporting Video")
            progress_dialog.setText("Exporting sliced video... Please wait.")
            progress_dialog.setStandardButtons(QMessageBox.NoButton)
            progress_dialog.show()
            
            QApplication.processEvents()
            
            while True:
                success, frame = self.video_capture.read()
                if not success:
                    break
                
                # Process frame to get layer images
                self.process_frame(frame)
                
                # Create combined frame
                combined = QPixmap(combined_width, max_height)
                combined.fill(Qt.black)
                
                painter = QPainter(combined)
                x_pos = 0
                for pixmap in self.layer_pixmaps:
                    painter.drawPixmap(x_pos, 0, pixmap)
                    x_pos += pixmap.width()
                painter.end()
                
                # Convert QPixmap to OpenCV image
                qimg = combined.toImage()
                ptr = qimg.constBits()
                ptr.setsize(qimg.byteCount())
                arr = np.array(ptr).reshape(max_height, combined_width, 4)  # RGBA
                bgr = cv2.cvtColor(arr, cv2.COLOR_RGBA2BGR)
                
                # Write frame
                writer.write(bgr)
                
                # Update progress (every 10 frames)
                if self.frame_count % 10 == 0:
                    percent = min(100, int(self.frame_count * 100 / frame_count))
                    progress_dialog.setText(f"Exporting sliced video: {percent}% complete")
                    QApplication.processEvents()
            
            # Release resources
            writer.release()
            
            progress_dialog.accept()
            
            self.log_debug(f"Exported sliced video to {file_path}")
            QMessageBox.information(self, "Export Complete", "Sliced video exported successfully.")
            
        except Exception as e:
            self.log_debug(f"ERROR exporting video: {str(e)}")
            QMessageBox.critical(self, "Error", f"Failed to export video: {str(e)}")

    def closeEvent(self, event):
        """Handle window close event"""
        if self.video_capture:
            self.video_capture.release()
        event.accept()

def main():
    app = QApplication(sys.argv)
    window = SliceViewConverter()
    window.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()