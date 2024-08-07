import numpy as np
from einops import rearrange

def getExtrinsic(camera_position, camera_direction):
    """ Return camera extrinsic where world coordinate is at [0,0,0] pointing at [0,0,1]
    """
    # Normalize the camera direction to get the forward vector
    forward = camera_direction / np.linalg.norm(camera_direction)

    # Define the world up vector
    world_up = np.array([0, 1, 0])
    right = np.cross(world_up, forward)
    right /= np.linalg.norm(right)
    up = np.cross(forward, right)
    up /= np.linalg.norm(up)
    rotation_matrix = np.array([right, up, forward])
    translation_vector = -np.dot(rotation_matrix, camera_position)
    
    # Create the extrinsic matrix
    extrinsic_matrix = np.eye(4)
    extrinsic_matrix[:3, :3] = rotation_matrix
    extrinsic_matrix[:3, 3] = translation_vector
    return extrinsic_matrix


def convertToCameraFrame(points, extrinsic):
    """ Convert n*3 numpy array of point cloud camera frame from given extrinsic
    """
    ones = np.ones((points.shape[0],1))
    points = np.hstack((points, ones)) # convert to homogenous coordinate
    transformed_points = points@(extrinsic.T)
    return transformed_points[:,0:3]
    


def projection(camera_frame_pts, image_width, image_height, num_pix_w, num_pix_h, focal_length):
    """ Project n*3 points onto a psudo image
    If multiple points are projected onto same pixel, only closest point will be kept
    """
    normal_image_points = camera_frame_pts[:,0:2] / rearrange(camera_frame_pts[:,-1], 'n -> n 1')
    image_points = normal_image_points * focal_length
    # apply translation in image, filter out points that are out of image after projected
    data = np.hstack((image_points, camera_frame_pts))


    data = data[data[:,-1].argsort()]
    data = data[data[:,1].argsort()]
    data = data[data[:,0].argsort()]
    _ , unique_indicies = np.unique(data[:,0:2], axis=0, return_index = True)
    data = data[unique_indicies]

    # "paint" points onto psuedo image
    image = np.zeros((num_pix_h, num_pix_w, 3))    
    for i in range(data.shape[0]):
        x = data[i, 0]
        y = data[i, 1]
        R, G, B = data[i,2], data[i,3], data[i,4]
        

    return image

def gaussianIntraPolation(filter_size):


    return


if __name__ == "__main__":
    data_path = r'/home/ansonhon/anson/thesis/LiDAR_NeRF/datasets/training_rainbow/round_plant2.npy'
    points = np.load(data_path)
    camera_position = np.array([0,0,0])
    camera_direction = np.array([0,0,1])

    camera_extrinsic = getExtrinsic(camera_position, camera_direction)
    camera_frame_pts = convertToCameraFrame(points, camera_extrinsic)
    psudo_image = projection(camera_frame_pts, 0.06, 0.06, 512, 512, 0.086148)




