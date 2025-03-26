import pygame
import numpy as np
import cv2
from PIL import Image

class VoxelFighter:
    def __init__(self, player, x, y, flip, data, sprite_sheet, animation_steps, sound):
        self.player = player
        self.size = data[0]
        self.image_scale = data[1]
        self.offset = data[2]
        self.flip = flip
        self.animation_steps = animation_steps
        
        # Create simple colored rectangles for different actions
        self.animation_list = self.create_dummy_animations()
        self.voxel_animations = self.convert_animations_to_voxel()
        
        self.action = 0
        self.frame_index = 0
        self.image = self.animation_list[self.action][self.frame_index]
        self.update_time = pygame.time.get_ticks()
        self.rect = pygame.Rect((x, y, 80, 180))
        self.vel_y = 0
        self.running = False
        self.jump = False
        self.attacking = False
        self.attack_type = 0
        self.attack_cooldown = 0
        self.attack_sound = sound
        self.hit = False
        self.health = 100
        self.alive = True
        
        # Voxel rendering parameters
        self.rotation_y = 0 if not flip else 180

    def create_dummy_animations(self):
        animation_list = []
        colors = [
            (255, 0, 0),    # Idle - Red
            (0, 255, 0),    # Run - Green
            (0, 0, 255),    # Jump - Blue
            (255, 255, 0),  # Attack1 - Yellow
            (255, 0, 255),  # Attack2 - Magenta
            (0, 255, 255),  # Hit - Cyan
            (128, 128, 128) # Death - Gray
        ]
        
        for i, steps in enumerate(self.animation_steps):
            temp_img_list = []
            for _ in range(steps):
                # Create a colored rectangle for each animation frame
                image = pygame.Surface((self.size, self.size), pygame.SRCALPHA)
                pygame.draw.rect(image, colors[i], (0, 0, self.size, self.size))
                scaled_image = pygame.transform.scale(image, 
                    (self.size * self.image_scale, self.size * self.image_scale))
                temp_img_list.append(scaled_image)
            animation_list.append(temp_img_list)
            
        return animation_list

    def convert_animations_to_voxel(self):
        voxel_animations = []
        for animation in self.animation_list:
            voxel_frames = []
            for frame in animation:
                # Convert Pygame surface to PIL Image
                frame_string = pygame.image.tostring(frame, 'RGBA')
                frame_pil = Image.frombytes('RGBA', frame.get_size(), frame_string)
                
                # Convert to numpy array
                frame_np = np.array(frame_pil)
                
                # Create depth map
                gray = cv2.cvtColor(cv2.cvtColor(frame_np, cv2.COLOR_RGBA2BGR), cv2.COLOR_BGR2GRAY)
                depth_map = cv2.GaussianBlur(gray, (3, 3), 0)
                
                # Edge enhancement
                edges = cv2.Canny(gray, 100, 200)
                depth_map = cv2.addWeighted(depth_map, 0.7, edges, 0.3, 0)
                
                voxel_frames.append({
                    'color_map': frame_np,
                    'depth_map': depth_map
                })
            voxel_animations.append(voxel_frames)
        return voxel_animations

    def rotate_point(self, point, rot_y):
        ry = np.radians(rot_y)
        x = point[0] * np.cos(ry) + point[2] * np.sin(ry)
        y = point[1]
        z = -point[0] * np.sin(ry) + point[2] * np.cos(ry)
        return [x, y, z]

    def render_voxel_frame(self, surface, camera_distance=1000):
        if not self.voxel_animations:
            return
            
        current_frame = self.voxel_animations[self.action][self.frame_index]
        depth_map = current_frame['depth_map']
        color_map = current_frame['color_map']
        
        height, width = depth_map.shape
        points = []
        
        # Convert 2D sprite to 3D points
        for y in range(height):
            for x in range(width):
                if color_map[y, x][3] > 0:  # Check alpha channel
                    depth = depth_map[y, x] / 255.0
                    
                    # Create 3D point
                    point = [
                        (x - width/2),
                        (y - height/2),
                        depth * 50  # Scale depth
                    ]
                    
                    # Apply rotation
                    rotated = self.rotate_point(point, self.rotation_y)
                    
                    # Project to 2D with perspective
                    scale = camera_distance / (camera_distance + rotated[2])
                    screen_x = int(rotated[0] * scale + self.rect.centerx)
                    screen_y = int(rotated[1] * scale + self.rect.centery)
                    
                    points.append((screen_x, screen_y, rotated[2], color_map[y, x][:3]))
        
        # Sort points by Z for proper depth ordering
        points.sort(key=lambda p: p[2], reverse=True)
        
        # Draw points
        for point in points:
            x, y, z, color = point
            if 0 <= x < surface.get_width() and 0 <= y < surface.get_height():
                size = max(1, int(2 * camera_distance/(camera_distance + z)))
                pygame.draw.rect(surface, color, (x, y, size, size))

    def move(self, screen_width, screen_height, surface, target, round_over):
        SPEED = 10
        GRAVITY = 2
        dx = 0
        dy = 0
        self.running = False
        self.attack_type = 0

        if self.attacking == False and self.alive == True and round_over == False:
            if self.player == 1:
                if pygame.key.get_pressed()[pygame.K_a]:
                    dx = -SPEED
                    self.running = True
                    self.rotation_y = 180
                if pygame.key.get_pressed()[pygame.K_d]:
                    dx = SPEED
                    self.running = True
                    self.rotation_y = 0
                if pygame.key.get_pressed()[pygame.K_w] and self.jump == False:
                    self.vel_y = -30
                    self.jump = True
                if pygame.key.get_pressed()[pygame.K_r] or pygame.key.get_pressed()[pygame.K_t]:
                    self.attack(target)
                    if pygame.key.get_pressed()[pygame.K_r]: self.attack_type = 1
                    if pygame.key.get_pressed()[pygame.K_t]: self.attack_type = 2

            if self.player == 2:
                if pygame.key.get_pressed()[pygame.K_LEFT]:
                    dx = -SPEED
                    self.running = True
                    self.rotation_y = 180
                if pygame.key.get_pressed()[pygame.K_RIGHT]:
                    dx = SPEED
                    self.running = True
                    self.rotation_y = 0
                if pygame.key.get_pressed()[pygame.K_UP] and self.jump == False:
                    self.vel_y = -30
                    self.jump = True
                if pygame.key.get_pressed()[pygame.K_KP1] or pygame.key.get_pressed()[pygame.K_KP2]:
                    self.attack(target)
                    if pygame.key.get_pressed()[pygame.K_KP1]: self.attack_type = 1
                    if pygame.key.get_pressed()[pygame.K_KP2]: self.attack_type = 2

        self.vel_y += GRAVITY
        dy += self.vel_y

        if self.rect.left + dx < 0:
            dx = -self.rect.left
        if self.rect.right + dx > screen_width:
            dx = screen_width - self.rect.right
        if self.rect.bottom + dy > screen_height - 110:
            self.vel_y = 0
            self.jump = False
            dy = screen_height - 110 - self.rect.bottom

        if target.rect.centerx > self.rect.centerx:
            self.flip = False
            self.rotation_y = 0
        else:
            self.flip = True
            self.rotation_y = 180

        if self.attack_cooldown > 0:
            self.attack_cooldown -= 1

        self.rect.x += dx
        self.rect.y += dy

    def update(self):
        if self.health <= 0:
            self.health = 0
            self.alive = False
            self.update_action(6)
        elif self.hit == True:
            self.update_action(5)
        elif self.attacking == True:
            if self.attack_type == 1:
                self.update_action(3)
            elif self.attack_type == 2:
                self.update_action(4)
        elif self.jump == True:
            self.update_action(2)
        elif self.running == True:
            self.update_action(1)
        else:
            self.update_action(0)

        animation_cooldown = 50
        if pygame.time.get_ticks() - self.update_time > animation_cooldown:
            self.frame_index += 1
            self.update_time = pygame.time.get_ticks()
        if self.frame_index >= len(self.animation_list[self.action]):
            if self.alive == False:
                self.frame_index = len(self.animation_list[self.action]) - 1
            else:
                self.frame_index = 0
                if self.action == 3 or self.action == 4:
                    self.attacking = False
                    self.attack_cooldown = 20
                if self.action == 5:
                    self.hit = False
                    self.attacking = False
                    self.attack_cooldown = 20

    def attack(self, target):
        if self.attack_cooldown == 0:
            self.attacking = True
            self.attack_sound.play()
            attacking_rect = pygame.Rect(
                self.rect.centerx - (2 * self.rect.width * self.flip),
                self.rect.y,
                2 * self.rect.width,
                self.rect.height
            )
            if attacking_rect.colliderect(target.rect):
                target.health -= 10
                target.hit = True

    def update_action(self, new_action):
        if new_action != self.action:
            self.action = new_action
            self.frame_index = 0
            self.update_time = pygame.time.get_ticks()

    def draw(self, surface):
        self.render_voxel_frame(surface)