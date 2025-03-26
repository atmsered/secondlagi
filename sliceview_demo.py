import pygame
import numpy as np
import sys

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
        self.layer_spacing = 50
        self.perspective_factor = 0.85

    def draw_demo_content(self):
        """Draw some demo content on each layer"""
        colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]  # Red, Green, Blue
        for i, slice in enumerate(self.slices):
            # Clear slice
            slice.fill((0, 0, 0, 0))
            
            # Draw a filled rectangle
            rect_size = 100 - (i * 20)  # Smaller rectangles for deeper layers
            x = (self.slice_width - rect_size) // 2
            y = (self.slice_height - rect_size) // 2
            pygame.draw.rect(slice, colors[i], (x, y, rect_size, rect_size))
            
            # Add text
            font = pygame.font.SysFont('Arial', 30)
            text = font.render(f'Layer {i+1}', True, colors[i])
            text_rect = text.get_rect(center=(self.slice_width//2, 50))
            slice.blit(text, text_rect)

    def combine_slices(self, screen):
        """Combine all slices with depth effect"""
        screen.fill((20, 20, 20))  # Dark gray background
        
        # Render from back to front
        for i in range(self.num_layers-1, -1, -1):
            slice_surface = self.slices[i]
            offset = int(i * self.layer_spacing)
            
            # Apply transparency based on depth
            alpha = 255 - (i * 40)
            slice_surface.set_alpha(alpha)
            
            # Calculate position with depth offset
            x = i * self.slice_width + offset
            screen.blit(slice_surface, (x, 0))

def main():
    # Initialize Pygame
    pygame.init()
    
    # Set up the display
    screen_width = 1000
    screen_height = 600
    screen = pygame.display.set_mode((screen_width, screen_height))
    pygame.display.set_caption("SliceView Demo")
    
    # Create SliceView renderer
    slice_renderer = SliceViewRenderer(screen_width, screen_height, num_layers=3)
    
    # Main game loop
    clock = pygame.time.Clock()
    running = True
    
    print("SliceView Demo Running!")
    print("Controls:")
    print("- Use LEFT/RIGHT arrow keys to adjust layer spacing")
    print("- Press ESC or close window to quit")
    
    while running:
        # Handle events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                    
        # Handle continuous key presses
        keys = pygame.key.get_pressed()
        if keys[pygame.K_LEFT]:
            slice_renderer.layer_spacing = max(0, slice_renderer.layer_spacing - 1)
        if keys[pygame.K_RIGHT]:
            slice_renderer.layer_spacing = min(100, slice_renderer.layer_spacing + 1)
        
        # Draw demo content
        slice_renderer.draw_demo_content()
        
        # Combine and display slices
        slice_renderer.combine_slices(screen)
        pygame.display.flip()
        
        # Cap the framerate
        clock.tick(60)

    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    main()