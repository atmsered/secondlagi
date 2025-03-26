import pygame
import numpy as np
from PIL import Image
import cv2

class SliceViewRenderer:
    def __init__(self, screen_width, screen_height, num_layers=3):
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.num_layers = num_layers
        
        # Calculate slice dimensions
        self.slice_width = screen_width // num_layers
        self.slice_height = screen_height
        
        # Create surfaces for each slice
        self.slices = [pygame.Surface((self.slice_width, self.slice_height), pygame.SRCALPHA) 
                      for _ in range(num_layers)]
        
        # Layer depth parameters
        self.layer_spacing = 50  # Depth spacing between layers
        self.perspective_factor = 0.85  # For depth scaling
        
    def get_layer_scale(self, layer_index):
        """Calculate scale factor based on layer depth"""
        return self.perspective_factor ** layer_index
    
    def render_slice(self, surface, slice_index, content):
        """Render content to a specific slice with perspective scaling"""
        scale = self.get_layer_scale(slice_index)
        scaled_width = int(self.slice_width * scale)
        scaled_height = int(self.slice_height * scale)
        
        # Scale content
        if content is not None:
            scaled_content = pygame.transform.scale(content, (scaled_width, scaled_height))
            # Center the scaled content
            x = (self.slice_width - scaled_width) // 2
            y = (self.slice_height - scaled_height) // 2
            surface.blit(scaled_content, (x, y))

    def create_depth_offset(self, layer_index):
        """Create parallax offset based on layer depth"""
        return int(layer_index * self.layer_spacing)

    def combine_slices(self, screen):
        """Combine all slices with depth effect"""
        screen.fill((0, 0, 0))  # Clear screen
        
        # Render from back to front
        for i in range(self.num_layers-1, -1, -1):
            slice_surface = self.slices[i]
            offset = self.create_depth_offset(i)
            
            # Apply transparency based on depth
            alpha = 255 - (i * 40)  # Deeper layers are more transparent
            slice_surface.set_alpha(alpha)
            
            # Calculate position with depth offset
            x = i * self.slice_width + offset
            screen.blit(slice_surface, (x, 0))