import open3d
from train import LiDAR_NeRF
from einops import repeat, rearrange
from numpy import cos, sin, array, sqrt, arctan2
from utility import printProgress, quickVizNumpy
import numpy as np
from register_from_slam import quat2RotationMatrix



# for quering model
def getPsudoImgVec(dir, img_num_pixel_edge = 800, fov_angle = 38.4):
    """ Given a direction vector as input, 
    return a set (img_height * img_width) of vectors 
    that points +- 38.4 degrees of input vector
    """
    fov_angle_rad = np.deg2rad(fov_angle)
    f = 0.5 / (np.tan(fov_angle_rad))
    x_hat = np.array([f])    # vector with focal length assuming image size is 1 meter * 1 meter
    x_hat = repeat(x_hat, '1 -> h w', h = img_num_pixel_edge, w = img_num_pixel_edge)
    temp = np.arange(-0.5,0.5,1/img_num_pixel_edge)
    y_hat, z_hat = np.meshgrid(temp,temp)
    breakpoint()

    # normalize them

    # figure out rotation matrix that rotate [1,0,0] to input direction

    # rotate the set of vector to align with input direction

    return 



def quat2RotationMatrix(q):
    """
    Convert a quaternion into a rotation matrix.
    Args:
    q (numpy.ndarray): A 4-element array representing the quaternion (w, x, y, z)
    Returns:
    numpy.ndarray: A 3x3 rotation matrix
    """
    w, x, y, z = q

    # Compute the rotation matrix elements
    R = np.array([
        [1 - 2*y*y - 2*z*z, 2*x*y - 2*z*w, 2*x*z + 2*y*w],
        [2*x*y + 2*z*w, 1 - 2*x*x - 2*z*z, 2*y*z - 2*x*w],
        [2*x*z - 2*y*w, 2*y*z + 2*x*w, 1 - 2*x*x - 2*y*y]
    ])
    return R



def getMask(points, camera_center, camera_direction, focal_length, image_height, image_width):

    # Normalize the camera direction to get the viewing direction
    camera_direction = camera_direction / np.linalg.norm(camera_direction)

    # Compute the right and up vectors for the camera coordinate system
    up_vector = np.array([0, 0, 1])
    right_vector = np.cross(camera_direction, up_vector)
    right_vector /= np.linalg.norm(right_vector)
    up_vector = np.cross(right_vector, camera_direction)
    up_vector /= np.linalg.norm(up_vector)

    # create rotation matrix, and transform points to camera coordinate
    R = np.vstack([right_vector, up_vector, -camera_direction])
    translated_points = points - camera_center
    camera_coords = np.dot(R, translated_points.T).T

    # Perspective projection
    projected_points = focal_length * (camera_coords[:, :2] / camera_coords[:, 2][:, np.newaxis])

    # Scale and shift points to fit the image dimensions
    projected_points[:, 0] = (projected_points[:, 0] + 1) * image_width / 2
    projected_points[:, 1] = (1 - projected_points[:, 1]) * image_height / 2

    # now write the points into the "canvas"
    canvas = np.zeros((image_height, image_width))
    for i in range(projected_points.shape[0]):
        y = 800 - int(projected_points[i,0])
        x = int(projected_points[i,1])
        if abs(x) < image_height and abs(y) < image_width:
            canvas[x,y] = 1
    
    return canvas











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

