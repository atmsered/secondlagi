import tkinter as tk
from tkinter import filedialog, ttk, messagebox
import pygame
import numpy as np
from PIL import Image
import math
import cv2
import os

class AIVoxelConverter:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("3D Voxel Converter with Rotation")
        self.root.geometry("1200x800")
        
        self.depth_map = None
        self.color_map = None
        self.screen = None
        self.pygame_initialized = False
        
        # 3D transformation parameters
        self.rotation_x = 0
        self.rotation_y = 0
        self.rotation_z = 0
        self.zoom = 1.0
        
        self.setup_gui()
        
    def setup_gui(self):
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Controls
        control_frame = ttk.Frame(main_frame)
        control_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # Load button
        ttk.Button(control_frame, text="Load Image", command=self.load_image).pack(side=tk.LEFT, padx=5)
        
        # Rotation controls
        rotation_frame = ttk.LabelFrame(control_frame, text="Rotation Controls", padding=5)
        rotation_frame.pack(side=tk.LEFT, padx=20)
        
        # X rotation
        ttk.Label(rotation_frame, text="X Rotation:").pack(side=tk.LEFT)
        self.x_rotation = ttk.Scale(rotation_frame, from_=0, to=360, orient=tk.HORIZONTAL)
        self.x_rotation.pack(side=tk.LEFT, padx=5)
        
        # Y rotation
        ttk.Label(rotation_frame, text="Y Rotation:").pack(side=tk.LEFT)
        self.y_rotation = ttk.Scale(rotation_frame, from_=0, to=360, orient=tk.HORIZONTAL)
        self.y_rotation.pack(side=tk.LEFT, padx=5)
        
        # Z rotation
        ttk.Label(rotation_frame, text="Z Rotation:").pack(side=tk.LEFT)
        self.z_rotation = ttk.Scale(rotation_frame, from_=0, to=360, orient=tk.HORIZONTAL)
        self.z_rotation.pack(side=tk.LEFT, padx=5)
        
        # Zoom control
        zoom_frame = ttk.LabelFrame(control_frame, text="Zoom", padding=5)
        zoom_frame.pack(side=tk.LEFT, padx=20)
        
        self.zoom_scale = ttk.Scale(zoom_frame, from_=0.1, to=2.0, orient=tk.HORIZONTAL)
        self.zoom_scale.set(1.0)
        self.zoom_scale.pack(side=tk.LEFT, padx=5)
        
        # Status label
        self.status_label = ttk.Label(control_frame, text="Status: Ready")
        self.status_label.pack(side=tk.LEFT, padx=20)
        
        # Display frame
        self.display_frame = ttk.Frame(main_frame)
        self.display_frame.pack(fill=tk.BOTH, expand=True)

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

    def load_image(self):
        try:
            file_path = filedialog.askopenfilename(
                filetypes=[("Image files", "*.png *.jpg *.jpeg *.bmp")]
            )
            
            if not file_path:
                return
                
            self.update_status("Loading image...")
            
            # Load and process image
            image = Image.open(file_path)
            image = image.convert('RGB')
            image = image.resize((128, 128))  # Smaller size for better performance
            
            # Convert to numpy array
            img_array = np.array(image)
            
            self.update_status("Processing depth map...")
            
            # Create depth map using multiple techniques
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
            self.depth_map = cv2.GaussianBlur(gray, (3, 3), 0)
            
            # Edge enhancement
            edges = cv2.Canny(gray, 100, 200)
            self.depth_map = cv2.addWeighted(self.depth_map, 0.7, edges, 0.3, 0)
            
            self.color_map = img_array
            
            if not self.pygame_initialized:
                self.init_pygame()
            
            self.update_status("Rendering...")
            self.render_loop()
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load image: {str(e)}")
            self.update_status("Error loading image")

    def init_pygame(self):
        pygame.init()
        self.display_frame.update()
        width = self.display_frame.winfo_width()
        height = self.display_frame.winfo_height()
        
        os.environ['SDL_WINDOWID'] = str(self.display_frame.winfo_id())
        self.screen = pygame.display.set_mode((width, height))
        pygame.display.init()
        
        self.pygame_initialized = True

    def render_voxels(self):
        if self.depth_map is None or self.color_map is None:
            return
            
        try:
            self.screen.fill((0, 0, 0))
            
            height, width = self.depth_map.shape
            screen_width = self.screen.get_width()
            screen_height = self.screen.get_height()
            
            # Get current rotation values
            rot_x = self.x_rotation.get()
            rot_y = self.y_rotation.get()
            rot_z = self.z_rotation.get()
            zoom = self.zoom_scale.get()
            
            # Center offset for rotation
            center_x = screen_width // 2
            center_y = screen_height // 2
            
            # Create points for each pixel
            points = []
            for y in range(height):
                for x in range(width):
                    if self.depth_map[y, x] > 30:  # Threshold to remove background
                        depth = self.depth_map[y, x] / 255.0
                        color = self.color_map[y, x]
                        
                        # Create 3D point
                        point = [
                            (x - width/2) * zoom,
                            (y - height/2) * zoom,
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
                if 0 <= point[0] < screen_width and 0 <= point[1] < screen_height:
                    size = int(2 * self.zoom_scale.get())
                    pygame.draw.rect(self.screen, point[3], (point[0], point[1], size, size))
            
            pygame.display.flip()
            
        except Exception as e:
            self.update_status(f"Render error: {str(e)}")

    def update_status(self, message):
        self.status_label.config(text=f"Status: {message}")
        self.root.update()

    def render_loop(self):
        clock = pygame.time.Clock()
        running = True
        
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
            
            self.render_voxels()
            self.root.update()
            clock.tick(30)

    def run(self):
        try:
            self.root.mainloop()
        finally:
            pygame.quit()

if __name__ == "__main__":
    app = AIVoxelConverter()
    app.run()