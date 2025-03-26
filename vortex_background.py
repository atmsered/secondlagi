import pygame
import numpy as np
import math

class VortexBackground:
    def __init__(self, screen_width, screen_height):
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.slices = []
        self.time = 0
        self.slice_count = 20
        self.colors = [
            (40, 0, 80),    # Dark purple
            (80, 0, 160),   # Purple
            (120, 0, 200),  # Bright purple
            (160, 0, 240)   # Light purple
        ]
        self.init_slices()

    def init_slices(self):
        for i in range(self.slice_count):
            angle = (i / self.slice_count) * 2 * math.pi
            self.slices.append({
                'angle': angle,
                'radius': 1000 + i * 50,
                'z': i * 100,
                'rotation': 0
            })

    def update(self):
        self.time += 0.02
        for slice in self.slices:
            # Update Z position (moving towards viewer)
            slice['z'] -= 5
            if slice['z'] < -500:
                slice['z'] = (self.slice_count - 1) * 100
            
            # Update rotation
            slice['rotation'] += 0.01
            
            # Update radius with wave effect
            base_radius = 1000 + (slice['z'] + 500) * 0.5
            slice['radius'] = base_radius + math.sin(self.time + slice['z'] * 0.01) * 100

    def draw(self, screen):
        # Fill background with dark color
        screen.fill((20, 0, 40))
        
        # Sort slices by Z for proper rendering
        sorted_slices = sorted(self.slices, key=lambda x: x['z'], reverse=True)
        
        center_x = self.screen_width // 2
        center_y = self.screen_height // 2
        
        for slice in sorted_slices:
            # Calculate size based on Z position
            z_factor = (slice['z'] + 1000) / 1500
            radius = slice['radius'] * z_factor
            
            # Calculate points for the slice
            points = []
            segments = 32
            for i in range(segments):
                angle = (i / segments) * 2 * math.pi + slice['rotation']
                x = center_x + math.cos(angle) * radius
                y = center_y + math.sin(angle) * radius * 0.5  # Elliptical effect
                points.append((x, y))
            
            # Draw slice
            if len(points) >= 3:
                # Choose color based on Z position
                color_index = int((slice['z'] + 500) / 200) % len(self.colors)
                color = self.colors[color_index]
                
                # Add alpha for depth effect
                alpha = min(255, max(0, int(255 * z_factor)))
                
                # Create surface for semi-transparent drawing
                surface = pygame.Surface((self.screen_width, self.screen_height), pygame.SRCALPHA)
                pygame.draw.polygon(surface, (*color, alpha), points)
                
                # Draw lines between segments for grid effect
                for i in range(len(points)):
                    start = points[i]
                    end = points[(i + 1) % len(points)]
                    pygame.draw.line(surface, (*color, alpha), start, end, 2)
                
                # Add additional grid lines
                for i in range(0, segments, 4):
                    angle = (i / segments) * 2 * math.pi + slice['rotation']
                    x = center_x + math.cos(angle) * radius
                    y = center_y + math.sin(angle) * radius * 0.5
                    pygame.draw.line(surface, (*color, alpha), (center_x, center_y), (x, y), 1)
                
                screen.blit(surface, (0, 0))