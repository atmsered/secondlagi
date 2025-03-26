import pygame
import numpy as np
from advanced_renderer import AIVoxelSliceRenderer

class Fighter:
    def __init__(self, x, y, fighter_type="warrior"):
        self.x = x
        self.y = y
        self.fighter_type = fighter_type
        self.health = 100
        self.jumping = False
        self.jump_velocity = 0
        self.gravity = 0.8
        self.speed = 5
        self.sprite = self.create_fighter_sprite()
        
    def create_fighter_sprite(self):
        sprite = pygame.Surface((64, 64), pygame.SRCALPHA)
        
        if self.fighter_type == "warrior":
            # Create pixel art warrior with sword
            colors = {
                'armor': (50, 100, 150),
                'sword': (200, 200, 200),
                'cape': (100, 50, 50),
                'skin': (255, 220, 180)
            }
            
            # Body
            pygame.draw.rect(sprite, colors['armor'], (20, 15, 24, 30))
            # Head
            pygame.draw.rect(sprite, colors['skin'], (24, 5, 16, 16))
            # Sword
            pygame.draw.rect(sprite, colors['sword'], (10, 20, 30, 4))
            # Cape
            pygame.draw.rect(sprite, colors['cape'], (38, 15, 10, 25))
            
        else:  # mage
            # Create pixel art mage with staff
            colors = {
                'robe': (70, 40, 120),
                'staff': (150, 100, 50),
                'magic': (200, 100, 255),
                'skin': (255, 220, 180)
            }
            
            # Robe
            pygame.draw.rect(sprite, colors['robe'], (20, 15, 24, 35))
            # Head
            pygame.draw.rect(sprite, colors['skin'], (24, 5, 16, 16))
            # Staff
            pygame.draw.rect(sprite, colors['staff'], (44, 10, 4, 40))
            # Magic orb
            pygame.draw.circle(sprite, colors['magic'], (46, 10), 6)
        
        return sprite

    def update(self, keys, screen_width):
        # Movement logic
        if self.fighter_type == "warrior":
            if keys[pygame.K_a]:
                self.x = max(0, self.x - self.speed)
            if keys[pygame.K_d]:
                self.x = min(screen_width - 64, self.x + self.speed)
            if not self.jumping and keys[pygame.K_w]:
                self.jumping = True
                self.jump_velocity = -15
        else:
            if keys[pygame.K_LEFT]:
                self.x = max(0, self.x - self.speed)
            if keys[pygame.K_RIGHT]:
                self.x = min(screen_width - 64, self.x + self.speed)
            if not self.jumping and keys[pygame.K_UP]:
                self.jumping = True
                self.jump_velocity = -15
                
        # Jump physics
        if self.jumping:
            self.y += self.jump_velocity
            self.jump_velocity += self.gravity
            
            if self.y > 400:  # Ground level
                self.y = 400
                self.jumping = False
                self.jump_velocity = 0

class DesertBackground:
    def __init__(self, width, height):
        self.surface = pygame.Surface((width, height))
        self.width = width
        self.height = height
        self.create_background()
        
    def create_background(self):
        # Sky gradient
        for y in range(self.height // 2):
            color = self.interpolate_color((135, 206, 235), (255, 200, 100), y / (self.height // 2))
            pygame.draw.line(self.surface, color, (0, y), (self.width, y))
            
        # Desert ground
        ground_color = (210, 180, 140)
        pygame.draw.rect(self.surface, ground_color, (0, self.height//2, self.width, self.height//2))
        
        # Draw mountains
        mountain_color = (139, 69, 19)
        points = [(0, self.height//2), (200, self.height//3), 
                 (400, self.height//2), (600, self.height//3),
                 (self.width, self.height//2)]
        pygame.draw.polygon(self.surface, mountain_color, points)
        
        # Add clouds
        cloud_color = (255, 255, 255, 128)
        for x in range(0, self.width, 200):
            self.draw_cloud(x, 100, cloud_color)
            
    def draw_cloud(self, x, y, color):
        cloud_surface = pygame.Surface((100, 50), pygame.SRCALPHA)
        pygame.draw.ellipse(cloud_surface, color, (0, 0, 60, 40))
        pygame.draw.ellipse(cloud_surface, color, (20, 10, 60, 40))
        pygame.draw.ellipse(cloud_surface, color, (40, 0, 60, 40))
        self.surface.blit(cloud_surface, (x, y))
        
    def interpolate_color(self, color1, color2, factor):
        return tuple(int(c1 + (c2 - c1) * factor) for c1, c2 in zip(color1, color2))

class Game:
    def __init__(self):
        pygame.init()
        self.screen_width = 1200
        self.screen_height = 600
        self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
        pygame.display.set_caption("Voxel Brawler")
        
        # Initialize renderers
        self.voxel_renderer = AIVoxelSliceRenderer(self.screen_width//2, self.screen_height)
        self.slice_renderer = AIVoxelSliceRenderer(self.screen_width//2, self.screen_height)
        
        # Create background
        self.background = DesertBackground(assets/images/background/background.jpg)
        
        # Create fighters
        self.warrior = Fighter(200, 400, "warrior")
        self.mage = Fighter(800, 400, "mage")
        
        # Initialize game state
        self.running = True
        self.camera_angle = 0
        self.clock = pygame.time.Clock()
        
    def update(self):
        keys = pygame.key.get_pressed()
        self.warrior.update(keys, self.screen_width)
        self.mage.update(keys, self.screen_width)
        self.camera_angle = (self.camera_angle + 0.5) % 360
        
    def draw(self):
        # Draw background
        self.screen.blit(self.background.surface, (0, 0))
        
        # Draw split screen divider
        pygame.draw.line(self.screen, (255, 255, 255), 
                        (self.screen_width//2, 0), 
                        (self.screen_width//2, self.screen_height), 2)
        
        # Draw voxel view
        self.voxel_renderer.process_sprite(self.warrior.sprite)
        self.voxel_renderer.render_scene(
            self.screen.subsurface((0, 0, self.screen_width//2, self.screen_height)),
            self.camera_angle
        )
        
        # Draw slice view
        self.slice_renderer.process_sprite(self.mage.sprite)
        self.slice_renderer.render_scene(
            self.screen.subsurface((self.screen_width//2, 0, self.screen_width//2, self.screen_height)),
            self.camera_angle
        )
        
        # Draw UI
        self.draw_ui()
        
        pygame.display.flip()
        
    def draw_ui(self):
        # Health bars
        warrior_health_rect = pygame.Rect(50, 20, 200 * (self.warrior.health/100), 20)
        mage_health_rect = pygame.Rect(self.screen_width - 250, 20, 200 * (self.mage.health/100), 20)
        
        pygame.draw.rect(self.screen, (255, 0, 0), warrior_health_rect)
        pygame.draw.rect(self.screen, (255, 0, 0), mage_health_rect)
        
        # Player labels
        font = pygame.font.Font(None, 36)
        p1_text = font.render("P1: " + str(self.warrior.health), True, (255, 0, 0))
        p2_text = font.render("P2: " + str(self.mage.health), True, (255, 0, 0))
        
        self.screen.blit(p1_text, (50, 50))
        self.screen.blit(p2_text, (self.screen_width - 250, 50))
        
    def run(self):
        while self.running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        self.running = False
            
            self.update()
            self.draw()
            self.clock.tick(60)

if __name__ == "__main__":
    game = Game()
    game.run()