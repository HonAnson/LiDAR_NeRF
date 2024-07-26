import open3d
import torch
from train import LiDAR_NeRF
from einops import repeat, rearrange
from numpy import cos, sin, array, sqrt, arctan2
import pandas as pd
from utility import printProgress
import numpy as np
from register_from_slam import quat2RotationMatrix

def getMask(points, camera_center, camera_direction, image_height = 800, image_width = 800, focal_length = 1):
        # Normalize the camera direction to get the viewing direction
    camera_direction = camera_direction / np.linalg.norm(camera_direction)

    # Compute the right and up vectors for the camera coordinate system
    up_vector = np.array([0, 1, 0])
    right_vector = np.cross(camera_direction, up_vector)
    right_vector /= np.linalg.norm(right_vector)
    up_vector = np.cross(right_vector, camera_direction)
    up_vector /= np.linalg.norm(up_vector)

    # Create the camera rotation matrix
    R = np.vstack([right_vector, up_vector, -camera_direction])

    # Translate points so the camera center is at the origin
    translated_points = points - camera_center

    # Rotate points into the camera coordinate system
    camera_coords = np.dot(R, translated_points.T).T

    # Perspective projection
    projected_points = focal_length * (camera_coords[:, :2] / camera_coords[:, 2][:, np.newaxis])

    # Scale and shift points to fit the image dimensions
    projected_points[:, 0] = (projected_points[:, 0] + 1) * image_width / 2
    projected_points[:, 1] = (1 - projected_points[:, 1]) * image_height / 2

    return projected_points











if __name__ == "__main__":
    # NOTE: camera points at [1,0,0] when unrotated
    model_path = r'local/models/version5_trial0.pth'
    output_path = r'local/visualize/v5t0.csv'
    
    position = array([0,0,0])
    direction = array([1,0,0])
    pcd = visualize(model_path, output_path, position, direction)



# def getUnitVectorfromImage(direction, focal_length, height = 1, width = 1, width_resolution = 1000, height_resolution = 1000):
    
#     # Initialize the output array
#     unit_vectors = np.zeros((1000000, 3))
    
#     # Compute the pixel coordinates
#     x_coords, y_coords = np.meshgrid(np.arange(height_resolution), np.arange(width_resolution))
#     x_coords = x_coords * (width / width_resolution)
#     y_coords = y_coords * (height / height_resolution)
    
#     # Flatten the coordinates
#     x_coords = x_coords.flatten()
#     y_coords = y_coords.flatten()

#     # Compute the direction vectors
#     for i in range(height_resolution*width_resolution):
#         x = (x_coords[i] - width / 2)
#         y = (y_coords[i] - height / 2)
#         z = focal_length
#         vec = np.array([x, y, z])
#         unit_vectors[i] = vec / np.linalg.norm(vec)
#     return unit_vectors

