import sys
import numpy as np
from PyQt5.QtMultimedia import QMediaPlayer, QMediaContent
from PyQt5.QtCore import QUrl, Qt, QTimer, QPoint, QSize, QRect
from PyQt5.QtWidgets import (QApplication, QMainWindow, QVBoxLayout, QHBoxLayout, 
                             QLabel, QPushButton, QSlider, QFileDialog, QWidget, 
                             QGroupBox, QMessageBox, QCheckBox, QTextEdit, QSizePolicy,
                             QComboBox)
from PyQt5.QtGui import QImage, QPixmap, QPainter, QColor, QResizeEvent, QPen, QBrush, QRadialGradient
import cv2
import math
import time
import mediapipe as mp
import os

class EnhancedVoxelConverter(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Smooth High-Detail Voxel 3D Converter")
        self.setGeometry(100, 100, 1200, 800)
        
        # Set window to be responsive and resizable
        self.setMinimumSize(800, 600)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        
        # Video processing attributes
        self.video_capture = None
        self.current_frame = None
        self.depth_map = None
        self.color_map = None
        self.is_playing = False
        self.frame_rate = 30  # Default FPS
        self.frame_count = 0
        self.processing_time = 0
        self.current_pixmap = None
        
        # Person segmentation attributes
        self.mp_selfie_segmentation = mp.solutions.selfie_segmentation
        self.selfie_segmentation = self.mp_selfie_segmentation.SelfieSegmentation(model_selection=1)
        self.person_mask = None
        
        # Performance metrics
        self.last_frame_time = 0
        self.fps_counter = 0
        self.fps = 0
        
        self.performance_mode = False  # Toggle for high performance mode
        self.frame_cache = {}  # Cache for processed frames
        self.last_render_time = 0
        self.target_fps = self.frame_rate  # Target FPS, initially set to video FPS
        self.skip_frames = 0  # Number of frames to skip
        self.sync_counter = 0  # For audio-video sync

        # Audio playback attributes
        self.media_player = QMediaPlayer()
        self.audio_enabled = True  # Default to audio enabled
        self.audio_volume = 50  # Default volume (0-100)

        # Create central widget and main layout
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        main_layout = QVBoxLayout()
        self.central_widget.setLayout(main_layout)
        
        # Create display area
        self.display_label = QLabel()
        self.display_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.display_label.setMinimumSize(640, 480)
        self.display_label.setAlignment(Qt.AlignCenter)
        self.display_label.setStyleSheet("background-color: black;")
        main_layout.addWidget(self.display_label, 1)
        
        # Add debug output area
        debug_group = QGroupBox("Debug Output")
        debug_layout = QVBoxLayout()
        debug_group.setLayout(debug_layout)
        
        self.debug_output = QTextEdit()
        self.debug_output.setReadOnly(True)
        self.debug_output.setMaximumHeight(100)
        debug_layout.addWidget(self.debug_output)
        
        main_layout.addWidget(debug_group)
        
        # Create control layout
        control_layout = QHBoxLayout()
        main_layout.addLayout(control_layout)
        
        # Left controls - File and playback
        left_controls = QVBoxLayout()
        
        file_layout = QHBoxLayout()
        
        # Load Video Button
        self.load_button = QPushButton("Load Video")
        self.load_button.clicked.connect(self.load_video)
        file_layout.addWidget(self.load_button)
        
        # Play/Pause Button
        self.play_button = QPushButton("Play")
        self.play_button.clicked.connect(self.toggle_playback)
        file_layout.addWidget(self.play_button)
        
        left_controls.addLayout(file_layout)
        
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
        
        left_controls.addWidget(rotation_group)
        
        # Zoom Slider
        zoom_group = QGroupBox("Zoom")
        zoom_layout = QVBoxLayout()
        zoom_group.setLayout(zoom_layout)
        
        self.zoom_slider = QSlider(Qt.Horizontal)
        self.zoom_slider.setRange(10, 200)  # 0.1 to 2.0
        self.zoom_slider.setValue(100)
        zoom_layout.addWidget(self.zoom_slider)
        
        left_controls.addWidget(zoom_group)
        
        control_layout.addLayout(left_controls)
        
        # Center controls - Rendering options
        render_group = QGroupBox("Rendering Options")
        render_layout = QVBoxLayout()
        render_group.setLayout(render_layout)
        
        # Rendering style
        style_layout = QHBoxLayout()
        style_layout.addWidget(QLabel("Render Style:"))
        self.style_combo = QComboBox()
        self.style_combo.addItems(["Smooth Voxels", "Blocky Voxels", "Gradient Voxels", "Dot Cloud"])
        self.style_combo.currentIndexChanged.connect(self.change_render_style)
        style_layout.addWidget(self.style_combo)
        render_layout.addLayout(style_layout)
        
        # Detail level
        detail_layout = QHBoxLayout()
        detail_layout.addWidget(QLabel("Detail Level:"))
        self.detail_slider = QSlider(Qt.Horizontal)
        self.detail_slider.setRange(1, 10)
        self.detail_slider.setValue(8)
        detail_layout.addWidget(self.detail_slider)
        render_layout.addLayout(detail_layout)
        
        # Element size
        size_layout = QHBoxLayout()
        size_layout.addWidget(QLabel("Element Size:"))
        self.size_slider = QSlider(Qt.Horizontal)
        self.size_slider.setRange(1, 20)
        self.size_slider.setValue(5)
        size_layout.addWidget(self.size_slider)
        render_layout.addLayout(size_layout)
        
        # Smoothing
        smoothing_layout = QHBoxLayout()
        smoothing_layout.addWidget(QLabel("Smoothing:"))
        self.smoothing_slider = QSlider(Qt.Horizontal)
        self.smoothing_slider.setRange(0, 100)
        self.smoothing_slider.setValue(80)
        smoothing_layout.addWidget(self.smoothing_slider)
        render_layout.addLayout(smoothing_layout)
        
        # Anti-aliasing
        aa_layout = QHBoxLayout()
        self.aa_checkbox = QCheckBox("Anti-aliasing")
        self.aa_checkbox.setChecked(True)
        aa_layout.addWidget(self.aa_checkbox)
        
        # Lighting effects
        self.lighting_checkbox = QCheckBox("Lighting Effects")
        self.lighting_checkbox.setChecked(True)
        aa_layout.addWidget(self.lighting_checkbox)
        
        render_layout.addLayout(aa_layout)
        
        control_layout.addWidget(render_group)
        
        # Right controls - Processing options
        options_group = QGroupBox("Processing Options")
        options_layout = QVBoxLayout()
        options_group.setLayout(options_layout)
        
        # Resolution control
        resolution_layout = QHBoxLayout()
        resolution_layout.addWidget(QLabel("Resolution:"))
        self.resolution_slider = QSlider(Qt.Horizontal)
        self.resolution_slider.setRange(32, 512)  # Range from low to high resolution
        self.resolution_slider.setValue(192)
        resolution_layout.addWidget(self.resolution_slider)
        options_layout.addLayout(resolution_layout)
        
        # Depth Method Options
        depth_method_layout = QHBoxLayout()
        
        # Person-based depth toggle
        self.person_checkbox = QCheckBox("Person Segmentation")
        self.person_checkbox.setChecked(True)
        depth_method_layout.addWidget(self.person_checkbox)
        
        # Edge enhancement toggle
        self.edge_checkbox = QCheckBox("Edge Enhancement")
        self.edge_checkbox.setChecked(True)
        depth_method_layout.addWidget(self.edge_checkbox)
        
        options_layout.addLayout(depth_method_layout)
        
        # Color Options
        color_layout = QHBoxLayout()
        
        # Original color toggle
        self.original_color_checkbox = QCheckBox("Original Colors")
        self.original_color_checkbox.setChecked(True)
        self.original_color_checkbox.stateChanged.connect(self.toggle_color_mode)
        color_layout.addWidget(self.original_color_checkbox)
        
        # Depth color toggle
        self.depth_color_checkbox = QCheckBox("Depth-Based Colors")
        self.depth_color_checkbox.setChecked(False)
        self.depth_color_checkbox.stateChanged.connect(self.toggle_color_mode)
        color_layout.addWidget(self.depth_color_checkbox)
        
        options_layout.addLayout(color_layout)
        
        # Depth threshold slider
        depth_layout = QHBoxLayout()
        depth_layout.addWidget(QLabel("Depth Threshold:"))
        self.depth_slider = QSlider(Qt.Horizontal)
        self.depth_slider.setRange(0, 100)
        self.depth_slider.setValue(30)
        depth_layout.addWidget(self.depth_slider)
        options_layout.addLayout(depth_layout)
        
        # Depth range slider
        depth_range_layout = QHBoxLayout()
        depth_range_layout.addWidget(QLabel("Depth Range:"))
        self.depth_range_slider = QSlider(Qt.Horizontal)
        self.depth_range_slider.setRange(10, 200)
        self.depth_range_slider.setValue(100)
        depth_range_layout.addWidget(self.depth_range_slider)
        options_layout.addLayout(depth_range_layout)
        
        control_layout.addWidget(options_group)
        
        # Add Audio Controls
        audio_group = QGroupBox("Audio Controls")
        audio_layout = QVBoxLayout()
        audio_group.setLayout(audio_layout)

        # Audio enable checkbox
        audio_checkbox_layout = QHBoxLayout()
        self.audio_checkbox = QCheckBox("Enable Audio")
        self.audio_checkbox.setChecked(True)
        self.audio_checkbox.stateChanged.connect(self.toggle_audio)
        audio_checkbox_layout.addWidget(self.audio_checkbox)
        audio_layout.addLayout(audio_checkbox_layout)

        # Volume slider
        volume_layout = QHBoxLayout()
        volume_layout.addWidget(QLabel("Volume:"))
        self.volume_slider = QSlider(Qt.Horizontal)
        self.volume_slider.setRange(0, 100)
        self.volume_slider.setValue(50)
        self.volume_slider.valueChanged.connect(self.set_volume)
        volume_layout.addWidget(self.volume_slider)
        audio_layout.addLayout(volume_layout)

        control_layout.addWidget(audio_group)

        # Performance options
        performance_layout = QHBoxLayout()
        self.performance_checkbox = QCheckBox("High Performance Mode")
        self.performance_checkbox.setChecked(False)
        self.performance_checkbox.stateChanged.connect(self.toggle_performance_mode)
        performance_layout.addWidget(self.performance_checkbox)
        options_layout.addLayout(performance_layout)

        # FPS target slider
        fps_layout = QHBoxLayout()
        fps_layout.addWidget(QLabel("Target FPS:"))
        self.fps_slider = QSlider(Qt.Horizontal)
        self.fps_slider.setRange(10, 120)
        self.fps_slider.setValue(30)
        self.fps_slider.valueChanged.connect(self.update_target_fps)
        fps_layout.addWidget(self.fps_slider)
        options_layout.addLayout(fps_layout)

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
        
        # Install event filter to handle resize events
        self.installEventFilter(self)
        
        self.log_debug("Application initialized successfully")

    def change_render_style(self, index):
        """Change the rendering style based on combobox selection"""
        style_map = {
            0: "smooth",
            1: "blocky",
            2: "gradient",
            3: "dots"
        }
        
        self.render_style = style_map.get(index, "smooth")
        self.log_debug(f"Render style changed to: {self.render_style}")
        
        # Re-render if we have a frame
        if self.current_frame is not None and self.depth_map is not None:
            self.render_voxels()

    def eventFilter(self, obj, event):
        """Handle custom events including resize"""
        if obj is self and event.type() == QResizeEvent.Resize:
            # If we have a current pixmap, update it to fit the new display size
            if self.current_pixmap is not None:
                self.update_display()
        return super(EnhancedVoxelConverter, self).eventFilter(obj, event)

    def update_display(self):
        """Update display to fit the current window size"""
        if self.current_pixmap:
            # Resize the pixmap to fit the display while maintaining aspect ratio
            scaled_pixmap = self.current_pixmap.scaled(
                self.display_label.width(),
                self.display_label.height(),
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation
            )
            self.display_label.setPixmap(scaled_pixmap)

    def resizeEvent(self, event):
        """Handle window resize event"""
        super(EnhancedVoxelConverter, self).resizeEvent(event)
        
        # Update the display when resized
        self.update_display()
        
        # If we have a current frame, re-render it for the new size
        if self.current_frame is not None and self.depth_map is not None:
            self.render_voxels()

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

    def toggle_color_mode(self, state):
        """Handle mutual exclusivity of color mode checkboxes"""
        sender = self.sender()
        
        if sender == self.original_color_checkbox and state == Qt.Checked:
            self.depth_color_checkbox.setChecked(False)
        elif sender == self.depth_color_checkbox and state == Qt.Checked:
            self.original_color_checkbox.setChecked(False)
            
        # Ensure at least one is checked
        if not self.original_color_checkbox.isChecked() and not self.depth_color_checkbox.isChecked():
            if sender == self.original_color_checkbox:
                self.depth_color_checkbox.setChecked(True)
            else:
                self.original_color_checkbox.setChecked(True)
        
        # If we have a current frame, reprocess to show the color changes
        if self.current_frame is not None and self.depth_map is not None:
            self.render_voxels()

    def calculate_fps(self):
        """Calculate and display current frames per second"""
        self.fps = self.fps_counter
        self.fps_counter = 0
        if self.is_playing:
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
            
            # Set up audio playback
            media_content = QMediaContent(QUrl.fromLocalFile(file_path))
            self.media_player.setMedia(media_content)
            self.media_player.setVolume(self.audio_volume)
            
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
            if self.audio_enabled:
                self.media_player.pause()
            self.play_button.setText("Play")
            self.status_label.setText("Playback paused")
            self.log_debug("Playback paused")
        else:
            # Play
            # Calculate timer interval from frame rate (milliseconds)
            interval = int(1000 / self.target_fps)
            self.timer.start(interval)
            
            # Start audio if enabled
            if self.audio_enabled:
                # Get current video position in milliseconds
                video_pos = self.video_capture.get(cv2.CAP_PROP_POS_MSEC)
                self.media_player.setPosition(int(video_pos))
                self.media_player.play()
                
            self.play_button.setText("Pause")
            self.status_label.setText("Playing video")
            self.log_debug(f"Playback started with interval: {interval}ms ({self.target_fps} FPS)")
        
        self.is_playing = not self.is_playing

    def update_frame(self):
        if not self.video_capture:
            self.log_debug("WARNING: Attempted to update frame with no video loaded")
            return
        
        # Check if we should skip this frame
        if self.skip_frames > 0:
            self.skip_frames -= 1
            # Just read and discard frame
            success, _ = self.video_capture.read()
            if not success:
                self.handle_end_of_video()
            self.fps_counter += 1
            return
        
        # Measure frame processing time
        start_time = time.time()
        
        # Read next frame
        success, frame = self.video_capture.read()
        
        if not success:
            self.handle_end_of_video()
            return
        
        # Process frame
        self.current_frame = frame
        self.frame_count += 1
        
        # Process the frame with performance considerations
        self.process_frame(frame)
        
        # Calculate and record processing time
        end_time = time.time()
        self.processing_time = (end_time - start_time) * 1000  # Convert to ms
        
        # Determine if we need to skip frames
        if self.performance_mode and self.processing_time > (1000 / self.target_fps):
            # Calculate how many frames to skip to maintain target FPS
            self.skip_frames = int(self.processing_time / (1000 / self.target_fps))
            self.log_debug(f"Processing too slow ({self.processing_time:.2f}ms), skipping {self.skip_frames} frames")
        
        # Sync audio with video periodically
        self.sync_counter += 1
        if self.audio_enabled and self.is_playing and self.sync_counter >= 30:
            self.sync_counter = 0
            video_pos = self.video_capture.get(cv2.CAP_PROP_POS_MSEC)
            audio_pos = self.media_player.position()
            
            # If drift is more than 500ms, resync
            if abs(video_pos - audio_pos) > 500:
                self.log_debug(f"Resyncing audio: video={video_pos:.2f}ms, audio={audio_pos}ms")
                self.media_player.setPosition(int(video_pos))
        
        self.fps_counter += 1

    def segment_person(self, frame):
        """Segment person from the background using MediaPipe"""
        if not self.person_checkbox.isChecked():
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
                
                # Apply Gaussian blur to smooth the mask edges
                smoothing = self.smoothing_slider.value() / 100.0
                if smoothing > 0:
                    blur_amount = int(5 + smoothing * 10)
                    # Make sure blur amount is odd
                    if blur_amount % 2 == 0:
                        blur_amount += 1
                    mask = cv2.GaussianBlur(mask, (blur_amount, blur_amount), 0)
                
                return mask
            
            return None
        except Exception as e:
            self.log_debug(f"ERROR in person segmentation: {str(e)}")
            return None

    def process_frame(self, frame):
        # Get current resolution and parameters
        resolution = self.resolution_slider.value()
        detail_level = self.detail_slider.value()
        smoothing = self.smoothing_slider.value()
        edge_enabled = self.edge_checkbox.isChecked()
        person_enabled = self.person_checkbox.isChecked()

        # Create cache key based on parameters
        cache_key = f"{resolution}_{detail_level}_{smoothing}_{edge_enabled}_{person_enabled}"
        frame_hash = hash(frame.tobytes())
        full_key = f"{cache_key}_{frame_hash}"

        # Check if we have this frame in cache
        if self.performance_mode and full_key in self.frame_cache:
            self.log_debug("Using cached frame")
            self.depth_map, self.color_map, self.person_mask = self.frame_cache[full_key]
            self.render_voxels()
            return

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
            
            # Create depth map
            gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
            
            # Apply bilateral filter for smoother depth transitions while preserving edges
            smoothing = self.smoothing_slider.value() / 100.0
            if smoothing > 0:
                d = int(5 + smoothing * 10)  # Diameter of each pixel neighborhood
                sigma_color = 75  # Filter sigma in the color space
                sigma_space = 75  # Filter sigma in the coordinate space
                gray = cv2.bilateralFilter(gray, d, sigma_color, sigma_space)
            
            self.depth_map = gray
            
            # Segment person if enabled
            self.person_mask = self.segment_person(resized)
            
            # If we have a person mask, use it to modify the depth map
            if self.person_mask is not None:
                # Make person appear closer (lower depth values)
                person_area = self.person_mask > 128
                
                # Create a distance map for background (further from the person = greater depth)
                if np.any(person_area):
                    bg_mask = 255 - self.person_mask
                    dist_transform = cv2.distanceTransform(bg_mask, cv2.DIST_L2, 5)
                    dist_transform = cv2.normalize(dist_transform, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
                    
                    # Apply Gaussian blur to smooth the distance transform
                    if smoothing > 0:
                        blur_amount = int(5 + smoothing * 15)
                        if blur_amount % 2 == 0:
                            blur_amount += 1
                        dist_transform = cv2.GaussianBlur(dist_transform, (blur_amount, blur_amount), 0)
                    
                    # Create depth map that incorporates person segmentation
                    # Person in foreground, background elements at varying depths
                    modified_depth = self.depth_map.copy()
                    modified_depth[person_area] = self.depth_map[person_area] * 0.3  # Make person closer (lower depth)
                    
                    # Background gets deeper based on distance from person
                    bg_area = ~person_area
                    if np.any(bg_area):
                        # Add distance transform to background depth
                        scale_factor = self.depth_range_slider.value() / 100.0
                        modified_depth[bg_area] = np.maximum(
                            self.depth_map[bg_area], 
                            dist_transform[bg_area] * scale_factor
                        )
                    
                    self.depth_map = modified_depth
            
            # Apply edge enhancement if enabled
            if self.edge_checkbox.isChecked():
                edges = cv2.Canny(gray, 100, 200)
                # Dilate edges slightly to make them more visible
                edges = cv2.dilate(edges, None)
                # Apply Gaussian blur to smooth the edges
                if smoothing > 0:
                    blur_amount = int(3 + smoothing * 5)
                    if blur_amount % 2 == 0:
                        blur_amount += 1
                    edges = cv2.GaussianBlur(edges, (blur_amount, blur_amount), 0)
                
                self.depth_map = cv2.addWeighted(self.depth_map, 0.7, edges, 0.3, 0)
            
            # Convert to RGB for rendering
            self.color_map = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
            
            # Render voxel view
            self.render_voxels()

            if self.performance_mode:
                self.frame_cache[full_key] = (self.depth_map, self.color_map, self.person_mask)
                # Limit cache size
                if len(self.frame_cache) > 30:  # Keep only most recent 30 frames
                    oldest_key = next(iter(self.frame_cache))
                    del self.frame_cache[oldest_key]

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
        # In high performance mode, reduce number of elements rendered
        detail_level = self.detail_slider.value()
        if self.performance_mode:
            detail_level = max(1, detail_level - 3)  # Reduce detail level
            
        # Calculate step size based on detail level and performance mode
        step = max(1, 11 - detail_level)
        
        if self.depth_map is None or self.color_map is None:
            return
        
        try:
            # Enable anti-aliasing if checked
            use_antialiasing = self.aa_checkbox.isChecked()
            use_lighting = self.lighting_checkbox.isChecked()
            
            # Create a pixmap to draw on
            width = self.display_label.width()
            height = self.display_label.height()
            pixmap = QPixmap(width, height)
            pixmap.fill(Qt.black)
            
            # Create a painter
            painter = QPainter(pixmap)
            
            # Enable antialiasing if requested
            if use_antialiasing:
                painter.setRenderHint(QPainter.Antialiasing)
                painter.setRenderHint(QPainter.SmoothPixmapTransform)
            
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
            
            # Element size and detail level
            element_size = self.size_slider.value()
            detail_level = self.detail_slider.value()
            
            # Calculate step size based on detail level (higher detail = smaller step size)
            step = max(1, 11 - detail_level)
            
            # Collect elements (voxels/points)
            elements = []
            depth_min = 255
            depth_max = 0
            
            # First pass to determine depth range
            for y in range(0, img_height, step):
                for x in range(0, img_width, step):
                    if self.depth_map[y, x] > depth_threshold:  # Depth threshold
                        depth_val = self.depth_map[y, x]
                        depth_min = min(depth_min, depth_val)
                        depth_max = max(depth_max, depth_val)
            
            # Prevent division by zero
            if depth_max == depth_min:
                depth_max = depth_min + 1
            
            # Second pass to create voxels
            for y in range(0, img_height, step):
                for x in range(0, img_width, step):
                    depth_val = self.depth_map[y, x]
                    
                    if depth_val > depth_threshold:  # Depth threshold
                        # Calculate normalized depth (0-1)
                        norm_depth = (depth_val - depth_min) / (depth_max - depth_min)
                        
                        # Get original color
                        color = self.color_map[y, x]
                        
                        # Create 3D point with scaled depth
                        depth_scale = self.depth_range_slider.value() / 100.0
                        point = [
                            (x - img_width/2) * zoom,
                            (y - img_height/2) * zoom,
                            norm_depth * 100 * zoom * depth_scale  # Scale depth
                        ]
                        
                        # Apply rotation
                        rotated = self.rotate_point(point, rot_x, rot_y, rot_z)
                        
                        # Project to 2D with perspective
                        z_factor = 1000 / (1000 + rotated[2])  # Perspective factor
                        screen_x = int(rotated[0] * z_factor + center_x)
                        screen_y = int(rotated[1] * z_factor + center_y)
                        
                        # Store element data for rendering
                        elements.append((screen_x, screen_y, rotated[2], color, norm_depth, z_factor))
            
            # Sort elements by depth (back to front)
            elements.sort(key=lambda e: e[2], reverse=True)
            
            # Draw elements based on selected render style
            if self.render_style == "smooth":
                self.render_smooth_voxels(painter, elements, element_size, use_lighting)
            elif self.render_style == "blocky":
                self.render_blocky_voxels(painter, elements, element_size)
            elif self.render_style == "gradient":
                self.render_gradient_voxels(painter, elements, element_size, use_lighting)
            elif self.render_style == "dots":
                self.render_dot_cloud(painter, elements, element_size)
            
            # End painting
            painter.end()
            
            # Store the current pixmap for resizing
            self.current_pixmap = pixmap
            
            # Display the pixmap with proper scaling to fit the display area
            self.update_display()
            
        except Exception as e:
            self.log_debug(f"ERROR: Failed to render voxels: {str(e)}")
            self.status_label.setText(f"Render error: {str(e)}")

    def render_smooth_voxels(self, painter, elements, base_size, use_lighting):
        """Render smooth voxels using gradients for a more detailed look"""
        width = self.display_label.width()
        height = self.display_label.height()
        
        for element in elements:
            x, y, z, rgb_color, norm_depth, z_factor = element
            
            # Skip if outside display area
            if x < 0 or x >= width or y < 0 or y >= height:
                continue
            
            # Determine element color based on selected mode
            if self.original_color_checkbox.isChecked():
                # Use original color
                r, g, b = rgb_color
                base_color = QColor(int(r), int(g), int(b))
            else:
                # Use depth-based color (blue to red gradient)
                r = int(norm_depth * 255)
                g = int(150 - abs(norm_depth * 255 - 150))
                b = int(255 - norm_depth * 255)
                base_color = QColor(r, g, b)
            
            # Calculate size based on perspective (closer = larger)
            perspective_factor = 1 + (z_factor - 0.5) * 0.5
            size = max(1, int(base_size * perspective_factor))
            
            if use_lighting:
                # Create gradient for lighting effect
                gradient = QRadialGradient(x + size/2, y + size/2, size)
                
                # Center highlight color (brighter)
                highlight_color = QColor(
                    min(255, base_color.red() + 40),
                    min(255, base_color.green() + 40),
                    min(255, base_color.blue() + 40)
                )
                
                # Edge shadow color (darker)
                shadow_color = QColor(
                    max(0, base_color.red() - 40),
                    max(0, base_color.green() - 40),
                    max(0, base_color.blue() - 40)
                )
                
                gradient.setColorAt(0, highlight_color)
                gradient.setColorAt(0.7, base_color)
                gradient.setColorAt(1, shadow_color)
                
                # Create brush with gradient
                brush = QBrush(gradient)
                painter.setBrush(brush)
                # No outline for smoother appearance
                painter.setPen(Qt.NoPen)
            else:
                # Simple rendering without lighting
                painter.setBrush(base_color)
                painter.setPen(base_color)
            
            # Draw a rounded rectangle for smoother appearance
            painter.drawRoundedRect(x - size/2, y - size/2, size, size, size/4, size/4)

    def render_blocky_voxels(self, painter, elements, base_size):
        """Render blocky voxels (traditional voxel look)"""
        width = self.display_label.width()
        height = self.display_label.height()
        
        for element in elements:
            x, y, z, rgb_color, norm_depth, z_factor = element
            
            # Skip if outside display area
            if x < 0 or x >= width or y < 0 or y >= height:
                continue
            
            # Determine element color based on selected mode
            if self           
            self.timer.stop()
            if self.audio_enabled:
                self.media_player.pause()
            self.play_button.setText("Play")
            self.status_label.setText("Playback paused")
            self.log_debug("Playback paused")
        else:
            # Play
            # Calculate timer interval from frame rate (milliseconds)
            interval = int(1000 / self.target_fps)
            self.timer.start(interval)
            
            # Start audio if enabled
            if self.audio_enabled:
                # Get current video position in milliseconds
                video_pos = self.video_capture.get(cv2.CAP_PROP_POS_MSEC)
                self.media_player.setPosition(int(video_pos))
                self.media_player.play()
                
            self.play_button.setText("Pause")
            self.status_label.setText("Playing video")
            self.log_debug(f"Playback started with interval: {interval}ms ({self.target_fps} FPS)")
        
        self.is_playing = not self.is_playing

    def update_frame(self):
        if not self.video_capture:
            self.log_debug("WARNING: Attempted to update frame with no video loaded")
            return
        
        # Check if we should skip this frame
        if self.skip_frames > 0:
            self.skip_frames -= 1
            # Just read and discard frame
            success, _ = self.video_capture.read()
            if not success:
                self.handle_end_of_video()
            self.fps_counter += 1
            return
        
        # Measure frame processing time
        start_time = time.time()
        
        # Read next frame
        success, frame = self.video_capture.read()
        
        if not success:
            self.handle_end_of_video()
            return
        
        # Process frame
        self.current_frame = frame
        self.frame_count += 1
        
        # Process the frame with performance considerations
        self.process_frame(frame)
        
        # Calculate and record processing time
        end_time = time.time()
        self.processing_time = (end_time - start_time) * 1000  # Convert to ms
        
        # Determine if we need to skip frames
        if self.performance_mode and self.processing_time > (1000 / self.target_fps):
            # Calculate how many frames to skip to maintain target FPS
            self.skip_frames = int(self.processing_time / (1000 / self.target_fps))
            self.log_debug(f"Processing too slow ({self.processing_time:.2f}ms), skipping {self.skip_frames} frames")
        
        # Sync audio with video periodically
        self.sync_counter += 1
        if self.audio_enabled and self.is_playing and self.sync_counter >= 30:
            self.sync_counter = 0
            video_pos = self.video_capture.get(cv2.CAP_PROP_POS_MSEC)
            audio_pos = self.media_player.position()
            
            # If drift is more than 500ms, resync
            if abs(video_pos - audio_pos) > 500:
                self.log_debug(f"Resyncing audio: video={video_pos:.2f}ms, audio={audio_pos}ms")
                self.media_player.setPosition(int(video_pos))
        
        self.fps_counter += 1

    def segment_person(self, frame):
        """Segment person from the background using MediaPipe"""
        if not self.person_checkbox.isChecked():
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
                
                # Apply Gaussian blur to smooth the mask edges
                smoothing = self.smoothing_slider.value() / 100.0
                if smoothing > 0:
                    blur_amount = int(5 + smoothing * 10)
                    # Make sure blur amount is odd
                    if blur_amount % 2 == 0:
                        blur_amount += 1
                    mask = cv2.GaussianBlur(mask, (blur_amount, blur_amount), 0)
                
                return mask
            
            return None
        except Exception as e:
            self.log_debug(f"ERROR in person segmentation: {str(e)}")
            return None

    def process_frame(self, frame):
        # Get current resolution and parameters
        resolution = self.resolution_slider.value()
        detail_level = self.detail_slider.value()
        smoothing = self.smoothing_slider.value()
        edge_enabled = self.edge_checkbox.isChecked()
        person_enabled = self.person_checkbox.isChecked()

        # Create cache key based on parameters
        cache_key = f"{resolution}_{detail_level}_{smoothing}_{edge_enabled}_{person_enabled}"
        frame_hash = hash(frame.tobytes())
        full_key = f"{cache_key}_{frame_hash}"

        # Check if we have this frame in cache
        if self.performance_mode and full_key in self.frame_cache:
            self.log_debug("Using cached frame")
            self.depth_map, self.color_map, self.person_mask = self.frame_cache[full_key]
            self.render_voxels()
            return

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
            
            # Create depth map
            gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
            
            # Apply bilateral filter for smoother depth transitions while preserving edges
            smoothing = self.smoothing_slider.value() / 100.0
            if smoothing > 0:
                d = int(5 + smoothing * 10)  # Diameter of each pixel neighborhood
                sigma_color = 75  # Filter sigma in the color space
                sigma_space = 75  # Filter sigma in the coordinate space
                gray = cv2.bilateralFilter(gray, d, sigma_color, sigma_space)
            
            self.depth_map = gray
            
            # Segment person if enabled
            self.person_mask = self.segment_person(resized)
            
            # If we have a person mask, use it to modify the depth map
            if self.person_mask is not None:
                # Make person appear closer (lower depth values)
                person_area = self.person_mask > 128
                
                # Create a distance map for background (further from the person = greater depth)
                if np.any(person_area):
                    bg_mask = 255 - self.person_mask
                    dist_transform = cv2.distanceTransform(bg_mask, cv2.DIST_L2, 5)
                    dist_transform = cv2.normalize(dist_transform, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
                    
                    # Apply Gaussian blur to smooth the distance transform
                    if smoothing > 0:
                        blur_amount = int(5 + smoothing * 15)
                        if blur_amount % 2 == 0:
                            blur_amount += 1
                        dist_transform = cv2.GaussianBlur(dist_transform, (blur_amount, blur_amount), 0)
                    
                    # Create depth map that incorporates person segmentation
                    # Person in foreground, background elements at varying depths
                    modified_depth = self.depth_map.copy()
                    modified_depth[person_area] = self.depth_map[person_area] * 0.3  # Make person closer (lower depth)
                    
                    # Background gets deeper based on distance from person
                    bg_area = ~person_area
                    if np.any(bg_area):
                        # Add distance transform to background depth
                        scale_factor = self.depth_range_slider.value() / 100.0
                        modified_depth[bg_area] = np.maximum(
                            self.depth_map[bg_area], 
                            dist_transform[bg_area] * scale_factor
                        )
                    
                    self.depth_map = modified_depth
            
            # Apply edge enhancement if enabled
            if self.edge_checkbox.isChecked():
                edges = cv2.Canny(gray, 100, 200)
                # Dilate edges slightly to make them more visible
                edges = cv2.dilate(edges, None)
                # Apply Gaussian blur to smooth the edges
                if smoothing > 0:
                    blur_amount = int(3 + smoothing * 5)
                    if blur_amount % 2 == 0:
                        blur_amount += 1
                    edges = cv2.GaussianBlur(edges, (blur_amount, blur_amount), 0)
                
                self.depth_map = cv2.addWeighted(self.depth_map, 0.7, edges, 0.3, 0)
            
            # Convert to RGB for rendering
            self.color_map = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
            
            # Render voxel view
            self.render_voxels()

            if self.performance_mode:
                self.frame_cache[full_key] = (self.depth_map, self.color_map, self.person_mask)
                # Limit cache size
                if len(self.frame_cache) > 30:  # Keep only most recent 30 frames
                    oldest_key = next(iter(self.frame_cache))
                    del self.frame_cache[oldest_key]

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
        # In high performance mode, reduce number of elements rendered
        detail_level = self.detail_slider.value()
        if self.performance_mode:
            detail_level = max(1, detail_level - 3)  # Reduce detail level
            
        # Calculate step size based on detail level and performance mode
        step = max(1, 11 - detail_level)
        
        if self.depth_map is None or self.color_map is None:
            return
        
        try:
            # Enable anti-aliasing if checked
            use_antialiasing = self.aa_checkbox.isChecked()
            use_lighting = self.lighting_checkbox.isChecked()
            
            # Create a pixmap to draw on
            width = self.display_label.width()
            height = self.display_label.height()
            pixmap = QPixmap(width, height)
            pixmap.fill(Qt.black)
            
            # Create a painter
            painter = QPainter(pixmap)
            
            # Enable antialiasing if requested
            if use_antialiasing:
                painter.setRenderHint(QPainter.Antialiasing)
                painter.setRenderHint(QPainter.SmoothPixmapTransform)
            
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
            
            # Element size and detail level
            element_size = self.size_slider.value()
            detail_level = self.detail_slider.value()
            
            # Calculate step size based on detail level (higher detail = smaller step size)
            step = max(1, 11 - detail_level)
            
            # Collect elements (voxels/points)
            elements = []
            depth_min = 255
            depth_max = 0
            
            # First pass to determine depth range
            for y in range(0, img_height, step):
                for x in range(0, img_width, step):
                    if self.depth_map[y, x] > depth_threshold:  # Depth threshold
                        depth_val = self.depth_map[y, x]
                        depth_min = min(depth_min, depth_val)
                        depth_max = max(depth_max, depth_val)
            
            # Prevent division by zero
            if depth_max == depth_min:
                depth_max = depth_min + 1
            
            # Second pass to create voxels
            for y in range(0, img_height, step):
                for x in range(0, img_width, step):
                    depth_val = self.depth_map[y, x]
                    
                    if depth_val > depth_threshold:  # Depth threshold
                        # Calculate normalized depth (0-1)
                        norm_depth = (depth_val - depth_min) / (depth_max - depth_min)
                        
                        # Get original color
                        color = self.color_map[y, x]
                        
                        # Create 3D point with scaled depth
                        depth_scale = self.depth_range_slider.value() / 100.0
                        point = [
                            (x - img_width/2) * zoom,
                            (y - img_height/2) * zoom,
                            norm_depth * 100 * zoom * depth_scale  # Scale depth
                        ]
                        
                        # Apply rotation
                        rotated = self.rotate_point(point, rot_x, rot_y, rot_z)
                        
                        # Project to 2D with perspective
                        z_factor = 1000 / (1000 + rotated[2])  # Perspective factor
                        screen_x = int(rotated[0] * z_factor + center_x)
                        screen_y = int(rotated[1] * z_factor + center_y)
                        
                        # Store element data for rendering
                        elements.append((screen_x, screen_y, rotated[2], color, norm_depth, z_factor))
            
            # Sort elements by depth (back to front)
            elements.sort(key=lambda e: e[2], reverse=True)
            
            # Draw elements based on selected render style
            if self.render_style == "smooth":
                self.render_smooth_voxels(painter, elements, element_size, use_lighting)
            elif self.render_style == "blocky":
                self.render_blocky_voxels(painter, elements, element_size)
            elif self.render_style == "gradient":
                self.render_gradient_voxels(painter, elements, element_size, use_lighting)
            elif self.render_style == "dots":
                self.render_dot_cloud(painter, elements, element_size)
            
            # End painting
            painter.end()
            
            # Store the current pixmap for resizing
            self.current_pixmap = pixmap
            
            # Display the pixmap with proper scaling to fit the display area
            self.update_display()
            
        except Exception as e:
            self.log_debug(f"ERROR: Failed to render voxels: {str(e)}")
            self.status_label.setText(f"Render error: {str(e)}")

    def render_smooth_voxels(self, painter, elements, base_size, use_lighting):
        """Render smooth voxels using gradients for a more detailed look"""
        width = self.display_label.width()
        height = self.display_label.height()
        
        for element in elements:
            x, y, z, rgb_color, norm_depth, z_factor = element
            
            # Skip if outside display area
            if x < 0 or x >= width or y < 0 or y >= height:
                continue
            
            # Determine element color based on selected mode
            if self.original_color_checkbox.isChecked():
                # Use original color
                r, g, b = rgb_color
                base_color = QColor(int(r), int(g), int(b))
            else:
                # Use depth-based color (blue to red gradient)
                r = int(norm_depth * 255)
                g = int(150 - abs(norm_depth * 255 - 150))
                b = int(255 - norm_depth * 255)
                base_color = QColor(r, g, b)
            
            # Calculate size based on perspective (closer = larger)
            perspective_factor = 1 + (z_factor - 0.5) * 0.5
            size = max(1, int(base_size * perspective_factor))
            
            if use_lighting:
                # Create gradient for lighting effect
                gradient = QRadialGradient(x + size/2, y + size/2, size)
                
                # Center highlight color (brighter)
                highlight_color = QColor(
                    min(255, base_color.red() + 40),
                    min(255, base_color.green() + 40),
                    min(255, base_color.blue() + 40)
                )
                
                # Edge shadow color (darker)
                shadow_color = QColor(
                    max(0, base_color.red() - 40),
                    max(0, base_color.green() - 40),
                    max(0, base_color.blue() - 40)
                )
                
                gradient.setColorAt(0, highlight_color)
                gradient.setColorAt(0.7, base_color)
                gradient.setColorAt(1, shadow_color)
                
                # Create brush with gradient
                brush = QBrush(gradient)
                painter.setBrush(brush)
                # No outline for smoother appearance
                painter.setPen(Qt.NoPen)
            else:
                # Simple rendering without lighting
                painter.setBrush(base_color)
                painter.setPen(base_color)
            
            # Draw a rounded rectangle for smoother appearance
            painter.drawRoundedRect(x - size/2, y - size/2, size, size, size/4, size/4)

    def render_blocky_voxels(self, painter, elements, base_size):
        """Render blocky voxels (traditional voxel look)"""
        width = self.display_label.width()
        height = self.display_label.height()
        
        for element in elements:
            x, y, z, rgb_color, norm_depth, z_factor = element
            
            # Skip if outside display area
            if x < 0 or x >= width or y < 0 or y >= height:
                continue
            
            # Determine element color based on selected mode
                # Use original color
                r, g, b = rgb_color
                base_color = QColor(int(r), int(g), int(b))
            else:
                # Use depth-based color (blue to red gradient)
                r = int(norm_depth * 255)
                g = int(150 - abs(norm_depth * 255 - 150))
                b = int(255 - norm_depth * 255)
                base_color = QColor(r, g, b)
            
            # Calculate size based on perspective (closer = larger)
            perspective_factor = 1 + (z_factor - 0.5) * 0.5
            size = max(1, int(base_size * perspective_factor))
            
            # Draw a square voxel
            painter.setPen(Qt.NoPen)
            painter.setBrush(base_color)
            painter.drawRect(x - size/2, y - size/2, size, size)
            
            # Add outlines for blocky look
            painter.setPen(QColor(max(0, base_color.red() - 50), 
                                  max(0, base_color.green() - 50), 
                                  max(0, base_color.blue() - 50)))
            painter.drawLine(x - size/2, y - size/2, x + size/2, y - size/2)  # Top
            painter.drawLine(x - size/2, y - size/2, x - size/2, y + size/2)  # Left
            
            # Highlight edges for 3D effect
            painter.setPen(QColor(min(255, base_color.red() + 30), 
                                  min(255, base_color.green() + 30), 
                                  min(255, base_color.blue() + 30)))
            painter.drawLine(x + size/2, y - size/2, x + size/2, y + size/2)  # Right
            painter.drawLine(x - size/2, y + size/2, x + size/2, y + size/2)  # Bottom

    def render_gradient_voxels(self, painter, elements, base_size, use_lighting):
        """Render voxels with gradient colors based on depth for a smoother look"""
        width = self.display_label.width()
        height = self.display_label.height()
        
        for element in elements:
            x, y, z, rgb_color, norm_depth, z_factor = element
            
            # Skip if outside display area
            if x < 0 or x >= width or y < 0 or y >= height:
                continue
            
            # Get base color
            if self.original_color_checkbox.isChecked():
                r, g, b = rgb_color
                base_color = QColor(int(r), int(g), int(b))
            else:
                # Create smooth color gradient based on depth
                r = int(norm_depth * 255)
                g = int(150 - abs(norm_depth * 255 - 150))
                b = int(255 - norm_depth * 255)
                base_color = QColor(r, g, b)
            
            # Calculate size based on perspective (closer = larger)
            perspective_factor = 1 + (z_factor - 0.5) * 0.5
            size = max(1, int(base_size * perspective_factor))
            
            if use_lighting:
                # Create more complex gradient for smoother 3D effect
                gradient = QRadialGradient(x + size/2, y + size/2, size * 1.5)
                
                # Center highlight color (brighter)
                highlight_color = QColor(
                    min(255, base_color.red() + 60),
                    min(255, base_color.green() + 60),
                    min(255, base_color.blue() + 60),
                    230  # Semi-transparent for blending
                )
                
                # Middle color
                mid_color = QColor(
                    base_color.red(),
                    base_color.green(),
                    base_color.blue(),
                    200  # Semi-transparent for blending
                )
                
                # Edge shadow color (darker)
                shadow_color = QColor(
                    max(0, base_color.red() - 60),
                    max(0, base_color.green() - 60),
                    max(0, base_color.blue() - 60),
                    170  # More transparent at edges
                )
                
                gradient.setColorAt(0, highlight_color)
                gradient.setColorAt(0.5, mid_color)
                gradient.setColorAt(1, shadow_color)
                
                painter.setBrush(QBrush(gradient))
                painter.setPen(Qt.NoPen)
            else:
                # Simple fill with base color
                painter.setBrush(base_color)
                painter.setPen(base_color)
            
            # Draw a circle for smoother appearance
            painter.drawEllipse(x - size/2, y - size/2, size, size)

    def render_dot_cloud(self, painter, elements, base_size):
        """Render as a point cloud with varying dot sizes based on depth"""
        width = self.display_label.width()
        height = self.display_label.height()
        
        for element in elements:
            x, y, z, rgb_color, norm_depth, z_factor = element
            
            # Skip if outside display area
            if x < 0 or x >= width or y < 0 or y >= height:
                continue
            
            # Determine element color based on selected mode
            if self.original_color_checkbox.isChecked():
                # Use original color
                r, g, b = rgb_color
                base_color = QColor(int(r), int(g), int(b))
            else:
                # Use depth-based color (blue to red gradient)
                r = int(norm_depth * 255)
                g = int(150 - abs(norm_depth * 255 - 150))
                b = int(255 - norm_depth * 255)
                base_color = QColor(r, g, b)
            
            # Smaller dots for a more detailed look
            perspective_factor = 1 + (z_factor - 0.5) * 0.5
            dot_size = max(1, int(base_size * perspective_factor * 0.7))
            
            # Draw a small circle
            painter.setPen(Qt.NoPen)
            painter.setBrush(base_color)
            painter.drawEllipse(x - dot_size/2, y - dot_size/2, dot_size, dot_size)

    def toggle_performance_mode(self, state):
        """Toggle high performance mode"""
        self.performance_mode = state == Qt.Checked
        if self.performance_mode:
            self.log_debug("High performance mode enabled")
        else:
            self.log_debug("High performance mode disabled")
        
        # Clear cache when changing modes
        self.frame_cache.clear()

    def update_target_fps(self):
        """Update target FPS from slider"""
        self.target_fps = self.fps_slider.value()
        if self.is_playing:
            # Update timer interval
            interval = int(1000 / self.target_fps)
            self.timer.setInterval(interval)
            self.log_debug(f"Target FPS updated to {self.target_fps}")

    def toggle_audio(self, state):
        """Toggle audio playback"""
        self.audio_enabled = state == Qt.Checked
        if self.audio_enabled:
            if self.is_playing:
                self.media_player.play()
            self.volume_slider.setEnabled(True)
            self.log_debug("Audio enabled")
        else:
            self.media_player.pause()
            self.volume_slider.setEnabled(False)
            self.log_debug("Audio disabled")

    def set_volume(self, value):
        """Set volume level"""
        self.audio_volume = value
        self.media_player.setVolume(value)
        self.log_debug(f"Volume set to {value}%")

    def handle_end_of_video(self):
        """Handle end of video playback"""
        self.timer.stop()
        if self.audio_enabled:
            self.media_player.stop()
        self.video_capture.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Rewind
        self.play_button.setText("Play")
        self.is_playing = False
        self.status_label.setText("End of video")
        self.log_debug("End of video reached, rewinding to start")

    def seek_video(self, position_msec):
        """Seek video to specified position in milliseconds"""
        if not self.video_capture:
            return
            
        # Seek video
        self.video_capture.set(cv2.CAP_PROP_POS_MSEC, position_msec)
        
        # Seek audio
        if self.audio_enabled:
            self.media_player.setPosition(int(position_msec))
        
        # Read and display new frame
        success, frame = self.video_capture.read()
        if success:
            self.current_frame = frame
            self.process_frame(frame)

    def closeEvent(self, event):
        # Clean up resources when closing
        self.log_debug("Application shutting down, cleaning up resources")
        if self.video_capture:
            self.video_capture.release()
        self.media_player.stop()
        event.accept()

def main():
    try:
        # Enable high DPI scaling
        QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
        QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)
        
        # Start application
        app = QApplication(sys.argv)
        window = EnhancedVoxelConverter()
        window.show()
        sys.exit(app.exec_())
    except Exception as e:
        print(f"Fatal error: {str(e)}")

if __name__ == "__main__":
    main()           