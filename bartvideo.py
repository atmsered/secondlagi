import sys
import numpy as np
from PyQt5.QtWidgets import (QApplication, QMainWindow, QVBoxLayout, QHBoxLayout, 
                             QLabel, QPushButton, QSlider, QFileDialog, QWidget, 
                             QGroupBox, QMessageBox, QCheckBox, QTextEdit)
from PyQt5.QtGui import QImage, QPixmap, QPainter, QColor
from PyQt5.QtCore import Qt, QTimer, QPoint
import cv2
import math
import time
import mediapipe as mp
import os

class Video3DVoxelConverter(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("3D Video Voxel Converter with Person Segmentation")
        self.setGeometry(100, 100, 1200, 800)
        
        # Video processing attributes
        self.video_capture = None
        self.current_frame = None
        self.depth_map = None
        self.color_map = None
        self.is_playing = False
        self.frame_rate = 30  # Default FPS
        self.frame_count = 0
        self.processing_time = 0
        
        # Person segmentation attributes
        self.mp_selfie_segmentation = mp.solutions.selfie_segmentation
        self.selfie_segmentation = self.mp_selfie_segmentation.SelfieSegmentation(model_selection=1)
        self.person_mask = None
        
        # Performance metrics
        self.last_frame_time = 0
        self.fps_counter = 0
        self.fps = 0
        
        # Create central widget and main layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout()
        central_widget.setLayout(main_layout)
        
        # Create display area
        self.display_label = QLabel()
        self.display_label.setMinimumSize(800, 600)
        main_layout.addWidget(self.display_label)
        
        # Add debug output area
        debug_group = QGroupBox("Debug Output")
        debug_layout = QVBoxLayout()
        debug_group.setLayout(debug_layout)
        
        self.debug_output = QTextEdit()
        self.debug_output.setReadOnly(True)
        self.debug_output.setMaximumHeight(150)
        debug_layout.addWidget(self.debug_output)
        
        main_layout.addWidget(debug_group)
        
        # Create control layout
        control_layout = QHBoxLayout()
        main_layout.addLayout(control_layout)
        
        # Load Video Button
        self.load_button = QPushButton("Load Video")
        self.load_button.clicked.connect(self.load_video)
        control_layout.addWidget(self.load_button)
        
        # Play/Pause Button
        self.play_button = QPushButton("Play")
        self.play_button.clicked.connect(self.toggle_playback)
        control_layout.addWidget(self.play_button)
        
        # Rotation Controls
        rotation_group = QGroupBox("Rotation Controls")
        rotation_layout = QHBoxLayout()
        rotation_group.setLayout(rotation_layout)
        
        # X Rotation Slider
        x_layout = QVBoxLayout()
        x_layout.addWidget(QLabel("X:"))
        self.x_rotation_slider = self.create_rotation_slider()
        x_layout.addWidget(self.x_rotation_slider)
        rotation_layout.addLayout(x_layout)
        
        # Y Rotation Slider
        y_layout = QVBoxLayout()
        y_layout.addWidget(QLabel("Y:"))
        self.y_rotation_slider = self.create_rotation_slider()
        y_layout.addWidget(self.y_rotation_slider)
        rotation_layout.addLayout(y_layout)
        
        # Z Rotation Slider
        z_layout = QVBoxLayout()
        z_layout.addWidget(QLabel("Z:"))
        self.z_rotation_slider = self.create_rotation_slider()
        z_layout.addWidget(self.z_rotation_slider)
        rotation_layout.addLayout(z_layout)
        
        control_layout.addWidget(rotation_group)
        
        # Zoom Slider
        zoom_group = QGroupBox("Zoom")
        zoom_layout = QVBoxLayout()
        zoom_group.setLayout(zoom_layout)
        
        self.zoom_slider = QSlider(Qt.Horizontal)
        self.zoom_slider.setRange(10, 200)  # 0.1 to 2.0
        self.zoom_slider.setValue(100)
        zoom_layout.addWidget(self.zoom_slider)
        
        control_layout.addWidget(zoom_group)
        
        # Processing options
        options_group = QGroupBox("Processing Options")
        options_layout = QVBoxLayout()
        options_group.setLayout(options_layout)
        
        # Resolution control
        resolution_layout = QHBoxLayout()
        resolution_layout.addWidget(QLabel("Resolution:"))
        self.resolution_slider = QSlider(Qt.Horizontal)
        self.resolution_slider.setRange(16, 128)  # Lower for better performance
        self.resolution_slider.setValue(64)
        resolution_layout.addWidget(self.resolution_slider)
        options_layout.addLayout(resolution_layout)
        
        # Person Segmentation toggle
        person_seg_layout = QHBoxLayout()
        self.person_seg_checkbox = QCheckBox("Person Segmentation")
        self.person_seg_checkbox.setChecked(True)
        person_seg_layout.addWidget(self.person_seg_checkbox)
        options_layout.addLayout(person_seg_layout)
        
        # Edge enhancement toggle
        edge_layout = QHBoxLayout()
        self.edge_checkbox = QCheckBox("Edge Enhancement")
        self.edge_checkbox.setChecked(True)
        edge_layout.addWidget(self.edge_checkbox)
        options_layout.addLayout(edge_layout)
        
        # Depth threshold
        depth_layout = QHBoxLayout()
        depth_layout.addWidget(QLabel("Depth Threshold:"))
        self.depth_slider = QSlider(Qt.Horizontal)
        self.depth_slider.setRange(0, 100)
        self.depth_slider.setValue(30)
        depth_layout.addWidget(self.depth_slider)
        options_layout.addLayout(depth_layout)
        
        control_layout.addWidget(options_group)
        
        # Status Label
        self.status_label = QLabel("Status: Ready")
        main_layout.addWidget(self.status_label)
        
        # Create a timer for video playback
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        
        # Create a timer for FPS calculation
        self.fps_timer = QTimer()
        self.fps_timer.timeout.connect(self.calculate_fps)
        self.fps_timer.start(1000)  # Update every second
        
        self.log_debug("Application initialized successfully")

    def log_debug(self, message):
        """Add debug message to the debug output area"""
        timestamp = time.strftime("%H:%M:%S")
        self.debug_output.append(f"[{timestamp}] {message}")
        
        # Auto-scroll to the bottom
        scrollbar = self.debug_output.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())

    def create_rotation_slider(self):
        slider = QSlider(Qt.Horizontal)
        slider.setRange(0, 360)
        slider.setValue(0)
        return slider

    def calculate_fps(self):
        """Calculate and display current frames per second"""
        self.fps = self.fps_counter
        self.fps_counter = 0
        self.log_debug(f"Current FPS: {self.fps}, Processing time per frame: {self.processing_time:.2f}ms")

    def load_video(self):
        try:
            self.log_debug("Opening file dialog to select video")
            # Open file dialog
            file_path, _ = QFileDialog.getOpenFileName(
                self, "Open Video", "", "Video Files (*.mp4 *.avi *.mov *.mkv)"
            )
            
            if not file_path:
                self.log_debug("No file selected")
                return
            
            # Update status
            self.status_label.setText(f"Loading video: {file_path}")
            self.log_debug(f"Loading video file: {file_path}")
            
            # Close any existing video
            if self.video_capture is not None:
                self.log_debug("Closing previous video")
                self.video_capture.release()
            
            # Open video file
            self.video_capture = cv2.VideoCapture(file_path)
            
            if not self.video_capture.isOpened():
                error_msg = "Failed to open video file"
                self.log_debug(f"ERROR: {error_msg}")
                QMessageBox.critical(self, "Error", error_msg)
                self.status_label.setText("Error loading video")
                return
            
            # Get video properties
            self.frame_rate = self.video_capture.get(cv2.CAP_PROP_FPS)
            frame_count = int(self.video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
            frame_width = int(self.video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
            frame_height = int(self.video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            self.log_debug(f"Video properties: {frame_width}x{frame_height}, {self.frame_rate} FPS, {frame_count} frames")
            
            if self.frame_rate <= 0:
                self.frame_rate = 30  # Default if unable to determine
                self.log_debug(f"Invalid frame rate detected, using default: {self.frame_rate} FPS")
            
            # Read first frame
            success, frame = self.video_capture.read()
            if not success:
                error_msg = "Failed to read video frame"
                self.log_debug(f"ERROR: {error_msg}")
                QMessageBox.critical(self, "Error", error_msg)
                self.status_label.setText("Error reading video frame")
                return
            
            self.log_debug(f"First frame read successfully: shape={frame.shape}")
            
            # Process first frame
            self.current_frame = frame
            self.process_frame(frame)
            
            # Update status
            self.status_label.setText("Video loaded successfully. Press Play to start.")
            self.play_button.setText("Play")
            self.is_playing = False
            self.log_debug("Video loaded successfully, ready to play")
            
        except Exception as e:
            error_msg = f"Failed to load video: {str(e)}"
            self.log_debug(f"ERROR: {error_msg}")
            QMessageBox.critical(self, "Error", error_msg)
            self.status_label.setText("Error loading video")

    def toggle_playback(self):
        if not self.video_capture:
            error_msg = "No video loaded"
            self.log_debug(f"WARNING: {error_msg}")
            QMessageBox.warning(self, "Warning", error_msg)
            return
        
        if self.is_playing:
            # Pause
            self.timer.stop()
            self.play_button.setText("Play")
            self.status_label.setText("Playback paused")
            self.log_debug("Playback paused")
        else:
            # Play
            # Calculate timer interval from frame rate (milliseconds)
            interval = int(1000 / self.frame_rate)
            self.timer.start(interval)
            self.play_button.setText("Pause")
            self.status_label.setText("Playing video")
            self.log_debug(f"Playback started with interval: {interval}ms ({self.frame_rate} FPS)")
        
        self.is_playing = not self.is_playing

    def update_frame(self):
        if not self.video_capture:
            self.log_debug("WARNING: Attempted to update frame with no video loaded")
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
            self.log_debug("End of video reached, rewinding to start")
            return
        
        # Process frame
        self.current_frame = frame
        self.frame_count += 1
        
        self.process_frame(frame)
        
        # Calculate and record processing time
        end_time = time.time()
        self.processing_time = (end_time - start_time) * 1000  # Convert to ms
        self.fps_counter += 1

    def segment_person(self, frame):
        """Segment person from the background using MediaPipe."""
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
                mask = (results.segmentation_mask > 0.2).astype(np.uint8) * 255
                
                # Apply morphological operations to clean the mask
                kernel = np.ones((5, 5), np.uint8)
                mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
                mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
                
                return mask
            
            return None
        except Exception as e:
            self.log_debug(f"ERROR in person segmentation: {str(e)}")
            return None

    def process_frame(self, frame):
        try:
            # Get current resolution setting
            resolution = self.resolution_slider.value()
            
            # Resize for processing (maintain aspect ratio)
            h, w = frame.shape[:2]
            aspect = w / h
            if w > h:
                new_w = resolution
                new_h = int(resolution / aspect)
            else:
                new_h = resolution
                new_w = int(resolution * aspect)
            
            # Resize frame
            resized = cv2.resize(frame, (new_w, new_h))
            
            # Segment person if enabled
            self.person_mask = self.segment_person(resized)
            
            # Convert to RGB (from BGR)
            rgb_frame = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
            
            # Create depth map
            gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
            self.depth_map = cv2.GaussianBlur(gray, (3, 3), 0)
            
            # If we have a person mask, use it to modify the depth map
            if self.person_mask is not None:
                # Make person appear closer (lower depth values)
                person_area = self.person_mask > 128
                
                # Create a distance map for background (further from the person = greater depth)
                if np.any(person_area):
                    bg_mask = 255 - self.person_mask
                    dist_transform = cv2.distanceTransform(bg_mask, cv2.DIST_L2, 5)
                    dist_transform = cv2.normalize(dist_transform, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
                    
                    # Create depth map that incorporates person segmentation
                    # Person in foreground, background elements at varying depths
                    modified_depth = self.depth_map.copy()
                    modified_depth[person_area] = self.depth_map[person_area] * 0.5  # Make person closer (lower depth)
                    
                    # Background gets deeper based on distance from person
                    bg_area = ~person_area
                    if np.any(bg_area):
                        # Add distance transform to background depth
                        modified_depth[bg_area] = np.maximum(
                            self.depth_map[bg_area],
                            dist_transform[bg_area] * 0.7  # Weight for distance effect
                        )
                    
                    self.depth_map = modified_depth
            
            # Edge enhancement if enabled
            if self.edge_checkbox.isChecked():
                edges = cv2.Canny(gray, 100, 200)
                self.depth_map = cv2.addWeighted(self.depth_map, 0.7, edges, 0.3, 0)
            
            # Store color map
            self.color_map = rgb_frame
            
            # Render voxels
            self.render_voxels()
            
        except Exception as e:
            error_msg = f"Error processing frame: {str(e)}"
            self.log_debug(f"ERROR: {error_msg}")
            self.status_label.setText(error_msg)

    def rotate_point(self, point, rot_x, rot_y, rot_z):
        # Convert degrees to radians
        rx = math.radians(rot_x)
        ry = math.radians(rot_y)
        rz = math.radians(rot_z)
        
        # Rotation matrices
        # X rotation
        x = point[0]
        y = point[1] * math.cos(rx) - point[2] * math.sin(rx)
        z = point[1] * math.sin(rx) + point[2] * math.cos(rx)
        
        # Y rotation
        x_new = x * math.cos(ry) + z * math.sin(ry)
        y_new = y
        z_new = -x * math.sin(ry) + z * math.cos(ry)
        
        # Z rotation
        x_final = x_new * math.cos(rz) - y_new * math.sin(rz)
        y_final = x_new * math.sin(rz) + y_new * math.cos(rz)
        z_final = z_new
        
        return [x_final, y_final, z_final]

    def render_voxels(self):
        if self.depth_map is None or self.color_map is None:
            return
        
        try:
            # Create a pixmap to draw on
            width = self.display_label.width()
            height = self.display_label.height()
            pixmap = QPixmap(width, height)
            pixmap.fill(Qt.black)
            
            # Create a painter
            painter = QPainter(pixmap)
            
            # Get rotation and zoom values
            rot_x = self.x_rotation_slider.value()
            rot_y = self.y_rotation_slider.value()
            rot_z = self.z_rotation_slider.value()
            zoom = self.zoom_slider.value() / 100.0
            depth_threshold = self.depth_slider.value()
            
            # Image dimensions
            img_height, img_width = self.depth_map.shape
            
            # Center coordinates
            center_x = width // 2
            center_y = height // 2
            
            # Create a downsampling factor for better performance
            # Adjust this based on your performance needs
            downsample = max(1, int(img_width * img_height / 5000))
            
            # Collect points
            points = []
            
            # Person mask for segmentation-based depth
            has_person = self.person_mask is not None and self.person_seg_checkbox.isChecked()
            
            for y in range(0, img_height, downsample):
                for x in range(0, img_width, downsample):
                    if self.depth_map[y, x] > depth_threshold:  # Depth threshold
                        depth = self.depth_map[y, x] / 255.0
                        color = self.color_map[y, x]
                        
                        # Adjust depth based on segmentation
                        if has_person:
                            if self.person_mask[y, x] > 128:
                                # Person is in foreground
                                depth = depth * 0.5  # Make closer
                            else:
                                # Background is further back
                                dist_to_person = min(255, self.person_mask[y, x] + 50) / 255.0
                                depth = max(depth, dist_to_person)
                        
                        # Create 3D point
                        point = [
                            (x - img_width/2) * zoom,
                            (y - img_height/2) * zoom,
                            depth * 100 * zoom  # Scale depth
                        ]
                        
                        # Apply rotation
                        rotated = self.rotate_point(point, rot_x, rot_y, rot_z)
                        
                        # Project to 2D
                        scale = 1000 / (1000 + rotated[2])  # Perspective projection
                        screen_x = int(rotated[0] * scale + center_x)
                        screen_y = int(rotated[1] * scale + center_y)
                        
                        # Store point data for rendering
                        points.append((screen_x, screen_y, rotated[2], color))
            
            # Sort points by Z for proper depth ordering (paint back to front)
            points.sort(key=lambda p: p[2], reverse=True)
            
            # Draw points
            size = max(1, int(2 * zoom * downsample))
            for point in points:
                if 0 <= point[0] < width and 0 <= point[1] < height:
                    color = QColor(int(point[3][0]), int(point[3][1]), int(point[3][2]))
                    painter.setPen(color)
                    painter.setBrush(color)
                    painter.drawRect(point[0], point[1], size, size)
            
            # End painting
            painter.end()
            
            # Display the pixmap
            self.display_label.setPixmap(pixmap)
            
        except Exception as e:
            self.log_debug(f"ERROR: Failed to render voxels: {str(e)}")
            self.status_label.setText(f"Render error: {str(e)}")

    def closeEvent(self, event):
        # Clean up resources when closing
        self.log_debug("Application shutting down, cleaning up resources")
        if self.video_capture:
            self.video_capture.release()
        event.accept()

def main():
    try:
        # Start application
        app = QApplication(sys.argv)
        window = Video3DVoxelConverter()
        window.show()
        sys.exit(app.exec_())
    except Exception as e:
        print(f"Fatal error: {str(e)}")

if __name__ == "__main__":
    main()