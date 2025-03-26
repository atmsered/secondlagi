import sys
import numpy as np
from PyQt5.QtWidgets import (QApplication, QMainWindow, QVBoxLayout, QHBoxLayout, 
                             QLabel, QPushButton, QSlider, QFileDialog, QWidget, 
                             QGroupBox, QMessageBox)
from PyQt5.QtGui import QImage, QPixmap, QPainter, QColor
from PyQt5.QtCore import Qt, QPoint
from PIL import Image
import cv2
import math

class VoxelConverterApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("3D Voxel Converter with Rotation")
        self.setGeometry(100, 100, 1200, 800)
        
        # Image processing attributes
        self.depth_map = None
        self.color_map = None
        
        # Create central widget and main layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout()
        central_widget.setLayout(main_layout)
        
        # Create display area
        self.display_label = QLabel()
        self.display_label.setMinimumSize(800, 600)
        main_layout.addWidget(self.display_label)
        
        # Create control layout
        control_layout = QHBoxLayout()
        main_layout.addLayout(control_layout)
        
        # Load Image Button
        load_button = QPushButton("Load Image")
        load_button.clicked.connect(self.load_image)
        control_layout.addWidget(load_button)
        
        # Rotation Controls
        rotation_group = QGroupBox("Rotation Controls")
        rotation_layout = QHBoxLayout()
        rotation_group.setLayout(rotation_layout)
        
        # X Rotation Slider
        self.x_rotation_slider = self.create_rotation_slider("X")
        rotation_layout.addWidget(QLabel("X:"))
        rotation_layout.addWidget(self.x_rotation_slider)
        
        # Y Rotation Slider
        self.y_rotation_slider = self.create_rotation_slider("Y")
        rotation_layout.addWidget(QLabel("Y:"))
        rotation_layout.addWidget(self.y_rotation_slider)
        
        # Z Rotation Slider
        self.z_rotation_slider = self.create_rotation_slider("Z")
        rotation_layout.addWidget(QLabel("Z:"))
        rotation_layout.addWidget(self.z_rotation_slider)
        
        control_layout.addWidget(rotation_group)
        
        # Zoom Slider
        zoom_group = QGroupBox("Zoom")
        zoom_layout = QHBoxLayout()
        zoom_group.setLayout(zoom_layout)
        
        self.zoom_slider = QSlider(Qt.Horizontal)
        self.zoom_slider.setRange(10, 200)  # 0.1 to 2.0
        self.zoom_slider.setValue(100)
        self.zoom_slider.valueChanged.connect(self.render_voxels)
        zoom_layout.addWidget(self.zoom_slider)
        
        control_layout.addWidget(zoom_group)
        
        # Status Label
        self.status_label = QLabel("Status: Ready")
        main_layout.addWidget(self.status_label)

    def create_rotation_slider(self, axis):
        slider = QSlider(Qt.Horizontal)
        slider.setRange(0, 360)
        slider.valueChanged.connect(self.render_voxels)
        return slider

    def load_image(self):
        try:
            # Open file dialog
            file_path, _ = QFileDialog.getOpenFileName(
                self, "Open Image", "", "Image Files (*.png *.jpg *.jpeg *.bmp)"
            )
            
            if not file_path:
                return
            
            # Update status
            self.status_label.setText(f"Loading image: {file_path}")
            
            # Load and process image
            image = Image.open(file_path)
            image = image.convert('RGB')
            image = image.resize((128, 128))  # Smaller size for better performance
            
            # Convert to numpy array
            img_array = np.array(image)
            
            # Create depth map
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
            self.depth_map = cv2.GaussianBlur(gray, (3, 3), 0)
            
            # Edge enhancement
            edges = cv2.Canny(gray, 100, 200)
            self.depth_map = cv2.addWeighted(self.depth_map, 0.7, edges, 0.3, 0)
            
            # Store color map
            self.color_map = img_array
            
            # Render voxels
            self.render_voxels()
            
            # Update status
            self.status_label.setText("Image loaded and rendered successfully")
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load image: {str(e)}")
            self.status_label.setText("Error loading image")

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
            
            # Image dimensions
            img_height, img_width = self.depth_map.shape
            
            # Center coordinates
            center_x = width // 2
            center_y = height // 2
            
            # Collect and sort points
            points = []
            for y in range(img_height):
                for x in range(img_width):
                    if self.depth_map[y, x] > 30:  # Threshold to remove background
                        depth = self.depth_map[y, x] / 255.0
                        color = self.color_map[y, x]
                        
                        # Create 3D point
                        point = [
                            (x - img_width/2) * zoom,
                            (y - img_height/2) * zoom,
                            depth * 100 * zoom
                        ]
                        
                        # Apply rotation
                        rotated = self.rotate_point(point, rot_x, rot_y, rot_z)
                        
                        # Project to 2D
                        scale = 1000 / (1000 + rotated[2])  # Perspective projection
                        screen_x = int(rotated[0] * scale + center_x)
                        screen_y = int(rotated[1] * scale + center_y)
                        
                        points.append((screen_x, screen_y, rotated[2], color))
            
            # Sort points by Z for proper depth ordering
            points.sort(key=lambda p: p[2], reverse=True)
            
            # Draw points
            for point in points:
                if 0 <= point[0] < width and 0 <= point[1] < height:
                    size = int(2 * zoom)
                    color = QColor(point[3][0], point[3][1], point[3][2])
                    painter.setPen(color)
                    painter.setBrush(color)
                    painter.drawRect(point[0], point[1], size, size)
            
            # End painting
            painter.end()
            
            # Display the pixmap
            self.display_label.setPixmap(pixmap)
            
        except Exception as e:
            QMessageBox.critical(self, "Render Error", f"Failed to render voxels: {str(e)}")

def main():
    app = QApplication(sys.argv)
    window = VoxelConverterApp()
    window.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()