import open3d
from train import LiDAR_NeRF
from einops import repeat, rearrange
from numpy import cos, sin, array, sqrt, arctan2
from utility import printProgress, quickVizNumpy
import numpy as np
from register_from_slam import quat2RotationMatrix
import cv2
import torch
import pandas as pd



# for quering model
def getPsudoImgVec(dir, pixel_to_meter, image_width, image_height, focal_length):
    """ Given a direction vector as input, 
    return a set (img_height * img_width) of vectors 
    that points +- 38.4 degrees of input vector
    NOTE: that vectors here are expressed in LiVOX camera frame
    """
    
    x_hat = np.array([focal_length])    # vector with focal length assuming image size is 1 meter * 1 meter
    x_hat = repeat(x_hat, '1 -> h w', h = image_height, w = image_width)

    half_h = (image_height / pixel_to_meter) / 2
    half_w = (image_width / pixel_to_meter) / 2

    height_grid = np.arange(half_h,-half_h,-1/pixel_to_meter)
    width_grid = np.arange(half_w,-half_w,-1/pixel_to_meter)

    y_hat, z_hat = np.meshgrid(height_grid,width_grid)

    #normalize them
    norm = np.sqrt(x_hat**2 + y_hat**2 + z_hat**2)
    x_hat /= norm
    y_hat /= norm
    z_hat /= norm

    vectors = np.array([x_hat, y_hat, z_hat])
    vectors = rearrange(vectors, 'd h w -> h w d')
    return vectors



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



def getMask(points, camera_center, camera_direction, focal_length, pixel_to_meter, image_width, image_height):
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
    # NOTE: camera coords: column 0, 1, 2 are x, y, z position in camera frame
    # x points to right of the image, y points up, and z points out of page
    camera_coords = np.dot(R, translated_points.T).T

    # Perspective projection
    projected_points = -focal_length * (camera_coords[:, :2] / camera_coords[:, 2][:, np.newaxis])

    # Project points from Film coordinate to pixel coordinate
    pixelated_points = (projected_points * pixel_to_meter).astype(int)

    # now write the points into the "canvas"
    canvas = np.zeros((image_height, image_width))
    for i in range(pixelated_points.shape[0]):

        x = pixelated_points[i,0] - image_width // 2
        y = (image_height - (pixelated_points[i,1] )) - image_height //2
        if abs(x) < image_height and abs(y) < image_width:
            canvas[y,x] = 1
    
    return canvas




if __name__ == "__main__":
    # NOTE: camera points at [1,0,0] when unrotated
    model_path = r'local/models/ver_euclidean_trial3.pth'
    output_path = r'local/visualize/ver_eucli_trial3.csv'
    data_path = r'datasets/training_euclidean/building.npy' # for mask 


    ### MODEL PARAMETERS ###
    HIDDEN_DIM = 512
    EMBEDDING_DIM_DIR = 8
    EMBEDDING_DIM_POS = 8

    with open(data_path, 'rb') as file:
        training_data_np = np.load(file)
    # get poitns for creating mask
    np.random.shuffle(training_data_np)
    points = training_data_np[0:1000000, 6:9] / 0.024429254132039887        # constant from dataset information
    camera_pos = np.array([0,0,0])
    camera_dir = np.array([1,0,0])

    FOCAL_LENGTH = 1.43580442   # unit = meter
    pixel_to_meter = 500
    image_height = 500
    image_width = 500

    mask = getMask(points, camera_pos, camera_dir, FOCAL_LENGTH, pixel_to_meter, image_height, image_width)
    vectors = getPsudoImgVec(camera_dir, pixel_to_meter, image_width, image_height, FOCAL_LENGTH)
    
    dir = np.array([[0,0,0]])
    for i in range(image_height):
        for j in range(image_width):
            if mask[i,j] != 0:
                dir = np.vstack((dir, vectors[i,j]))

    dir = dir[1:,:]
    # tile positions too             
    pos_np = repeat(camera_pos, 'd -> n d', n = dir.shape[0])
    pos = torch.tensor(pos_np)
    dir = torch.tensor(dir)

    # inference from model
    model_evel = LiDAR_NeRF(hidden_dim=HIDDEN_DIM, embedding_dim_dir=EMBEDDING_DIM_DIR, embedding_dim_pos=EMBEDDING_DIM_POS)
    model_evel.load_state_dict(torch.load(model_path))
    model_evel.eval(); # Set the model to inference mode

    with torch.no_grad():
        pred_points = model_evel(pos, dir)
    breakpoint()
    pts_np = pred_points.detach().numpy()
    pts_np /= 0.024429254132039887
    # save the points for visualize
    # Load dummy csv
    df_temp = pd.read_csv('local/visualize/dummy.csv')
    df_temp = df_temp.head(pts_np.shape[0])

    # write to dummy csv
    df_temp['X'] = pts_np[:,0]
    df_temp['Y'] = pts_np[:,1]
    df_temp['Z'] = pts_np[:,2]
    df_temp.to_csv(output_path, index=False)
    print("Data written to ", output_path)

    # visualize the mask
    matrix_normalized = (mask * 255).astype(np.uint8)
    cv2.imshow('Mask', matrix_normalized)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    


