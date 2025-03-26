import pygame
import numpy as np
import cv2
from scipy import ndimage

class AIVoxelSliceRenderer:
    def __init__(self, screen_width, screen_height, voxel_resolution=32):
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.voxel_resolution = voxel_resolution
        
        # Voxel grid for 3D representation
        self.voxel_grid = np.zeros((voxel_resolution, voxel_resolution, voxel_resolution))
        
        # Rendering parameters
        self.slice_count = 8  # Number of depth slices
        self.depth_scale = 2.0  # Controls depth perception intensity
        self.edge_threshold = 0.3  # Edge detection sensitivity
        self.blur_sigma = 1.5  # Smoothing factor
        
        # Initialize render buffers
        self.depth_buffer = np.zeros((screen_height, screen_width))
        self.color_buffer = np.zeros((screen_height, screen_width, 3))
        self.slices = [pygame.Surface((screen_width, screen_height), pygame.SRCALPHA) 
                      for _ in range(self.slice_count)]

    def generate_voxel_representation(self, sprite_surface):
        """Convert 2D sprite into voxel representation with AI-enhanced depth"""
        # Convert pygame surface to numpy array
        sprite_array = pygame.surfarray.array3d(sprite_surface)
        sprite_array = sprite_array.transpose(1, 0, 2)  # Correct orientation
        
        # Generate depth map using multiple techniques
        depth_map = self._generate_ai_depth_map(sprite_array)
        
        # Create voxel representation
        sprite_height, sprite_width = depth_map.shape
        for x in range(sprite_width):
            for y in range(sprite_height):
                depth = depth_map[y, x]
                if depth > 0:
                    # Map to voxel coordinates
                    vx = int(x * self.voxel_resolution / sprite_width)
                    vy = int(y * self.voxel_resolution / sprite_height)
                    vz = int(depth * (self.voxel_resolution-1) / 255)
                    
                    # Set voxel and its neighbors for volume
                    self._set_voxel_with_neighbors(vx, vy, vz, sprite_array[y, x])

    def _generate_ai_depth_map(self, image):
        """Generate depth map using OpenCV techniques"""
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image.copy()
            
        # Multi-scale edge detection
        edges = np.zeros_like(gray, dtype=float)
        
        # Use different blur levels for multi-scale edge detection
        for sigma in [1, 2, 4]:
            blurred = cv2.GaussianBlur(gray, (0, 0), sigma)
            edge = cv2.Canny(blurred, 50, 150)
            edges += edge.astype(float) * (1.0 / sigma)
            
        # Generate depth gradient
        gradient_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        gradient_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        gradient_mag = np.sqrt(gradient_x**2 + gradient_y**2)
        
        # Combine edge and gradient information
        depth = (edges * 0.6 + gradient_mag * 0.4)
        
        # Apply non-linear depth curve
        depth = np.power(depth / depth.max(), self.depth_scale) * 255
        
        # Smooth depth map
        depth = cv2.GaussianBlur(depth, (0, 0), self.blur_sigma)
        
        return depth.astype(np.uint8)

    def _set_voxel_with_neighbors(self, x, y, z, color):
        """Set voxel and its neighbors for smoother representation"""
        def safe_set(dx, dy, dz, value):
            nx, ny, nz = x + dx, y + dy, z + dz
            if (0 <= nx < self.voxel_resolution and 
                0 <= ny < self.voxel_resolution and 
                0 <= nz < self.voxel_resolution):
                self.voxel_grid[nx, ny, nz] = value

        # Set center voxel
        safe_set(0, 0, 0, 1.0)
        
        # Set immediate neighbors with reduced intensity
        for dx, dy, dz in [(0,1,0), (1,0,0), (0,0,1), 
                          (0,-1,0), (-1,0,0), (0,0,-1)]:
            safe_set(dx, dy, dz, 0.7)

    def render_slice(self, slice_index, camera_angle=0):
        """Render a single depth slice with perspective"""
        slice_surface = self.slices[slice_index]
        slice_surface.fill((0, 0, 0, 0))
        
        # Calculate slice depth
        depth = slice_index / (self.slice_count - 1)
        z_layer = int(depth * (self.voxel_resolution - 1))
        
        # Apply perspective transformation
        angle_rad = np.radians(camera_angle)
        cos_angle = np.cos(angle_rad)
        sin_angle = np.sin(angle_rad)
        
        # Render voxels in this slice
        for x in range(self.voxel_resolution):
            for y in range(self.voxel_resolution):
                if self.voxel_grid[x, y, z_layer] > 0:
                    # Apply perspective transformation
                    screen_x = int((x - self.voxel_resolution/2) * cos_angle - 
                                 (z_layer - self.voxel_resolution/2) * sin_angle +
                                 self.screen_width/2)
                    screen_y = int(y * (1 + depth * 0.5) + 
                                 self.screen_height * (1 - depth) * 0.5)
                    
                    # Draw voxel with depth-based size and alpha
                    size = int(10 * (1 - depth * 0.5))
                    alpha = int(255 * (1 - depth * 0.7))
                    color = (255, 255, 255, alpha)
                    
                    pygame.draw.rect(slice_surface, color,
                                   (screen_x - size//2, screen_y - size//2,
                                    size, size))

    def render_scene(self, screen, camera_angle=0):
        """Render complete scene with all slices"""
        screen.fill((0, 0, 0))
        
        # Render each slice
        for i in range(self.slice_count):
            self.render_slice(i, camera_angle)
            
        # Composite slices from back to front
        for i in range(self.slice_count):
            offset_x = int(i * 2 * np.cos(np.radians(camera_angle)))
            offset_y = int(i * 1.5)  # Vertical offset for depth
            screen.blit(self.slices[i], (offset_x, offset_y))

    def process_sprite(self, sprite_surface):
        """Process sprite into voxel representation and prepare for rendering"""
        # Clear previous voxel data
        self.voxel_grid.fill(0)
        
        # Generate voxel representation
        self.generate_voxel_representation(sprite_surface)
        
        # Prepare slices for rendering
        for i in range(self.slice_count):
            self.slices[i].fill((0, 0, 0, 0))