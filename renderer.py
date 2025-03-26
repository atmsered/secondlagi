import numpy as np
import cv2

def enhance_depth_map(depth_map):
    """Enhance depth map for better 3D effect"""
    # Normalize depth values
    depth_map = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX)
    
    # Apply bilateral filter for edge preservation
    depth_map = cv2.bilateralFilter(depth_map.astype(np.uint8), 9, 75, 75)
    
    # Enhance edges
    edges = cv2.Canny(depth_map, 50, 150)
    depth_map = cv2.addWeighted(depth_map, 0.7, edges, 0.3, 0)
    
    # Apply non-linear transformation for better depth perception
    depth_map = np.power(depth_map / 255.0, 0.5) * 255
    
    return depth_map.astype(np.uint8)

def create_voxel_points(color_map, depth_map, scale=1.0):
    """Convert 2D image and depth map to 3D points"""
    height, width = depth_map.shape
    points = []
    
    for y in range(height):
        for x in range(width):
            if color_map[y, x][3] > 0:  # Check alpha channel
                depth = depth_map[y, x] / 255.0
                
                # Create point with color and position
                point = {
                    'position': np.array([
                        (x - width/2) * scale,
                        (y - height/2) * scale,
                        depth * 50 * scale
                    ]),
                    'color': color_map[y, x][:3]
                }
                points.append(point)
    
    return points

def rotate_points(points, angle_y):
    """Rotate points around Y axis"""
    rotation_matrix = np.array([
        [np.cos(angle_y), 0, np.sin(angle_y)],
        [0, 1, 0],
        [-np.sin(angle_y), 0, np.cos(angle_y)]
    ])
    
    rotated_points = []
    for point in points:
        rotated_position = rotation_matrix @ point['position']
        rotated_points.append({
            'position': rotated_position,
            'color': point['color']
        })
    
    return rotated_points

def project_points(points, camera_distance, screen_center):
    """Project 3D points to 2D screen space with perspective"""
    projected_points = []
    
    for point in points:
        # Apply perspective projection
        z = point['position'][2] + camera_distance
        if z > 0:
            scale = camera_distance / z
            screen_x = int(point['position'][0] * scale + screen_center[0])
            screen_y = int(point['position'][1] * scale + screen_center[1])
            
            projected_points.append({
                'screen_pos': (screen_x, screen_y),
                'depth': z,
                'color': point['color']
            })
    
    # Sort by depth for proper rendering
    projected_points.sort(key=lambda p: p['depth'], reverse=True)
    return projected_points