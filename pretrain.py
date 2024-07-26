import json
import numpy as np
import os
from einops import rearrange
from utility import printProgress
from numpy import sqrt, arctan2, array
# import open3d as o3d

""" Preprocess data including:
1. Filter out points further than 5m (depend on datasets)
2. Find regsiters between frames
3. Obtain global coordinate (frame zero's coordinate) origins for each frame
4. Obtain view direction in global frame for each frame
5. Combine and save as one numpy array
"""


def loadDataFromRegisteredSlam(path):
    # load file of registered point cloud
    with open(path,'r') as file:
        data = json.load(file)
    return data


def cart2sph(pcd_array, pose_position):
    """ Convert a n*3 point cloud, 
    with camera centre at pose_position from cartesian coordinate in global frame
    to global algined spherical coordinate at camera frame
    
    """
    pcd_local_aligned = pcd_array - pose_position
    x, y, z = pcd_local_aligned[:,0], pcd_local_aligned[:,1], pcd_local_aligned[:,2]
    XsqPlusYsq = x**2 + y**2
    r = sqrt(XsqPlusYsq + z**2)
    elev = arctan2(z, sqrt(XsqPlusYsq))
    pan = arctan2(y, x)
    output = array([r, elev, pan])
    return rearrange(output, 'a b -> b a') #take transpose


def preProcess(data):
    """Transform registered point clouds into giant array for model training
    input: data of type dict, with keys = frames
    output: n*6 numpy array
    """
    keys = list(data.keys())
    output = np.zeros((1,6))
    iter = 0
    total_iter = len(keys)
    for key in keys:
        pcd_cart = np.array(data[key]['point_cloud'])
        pose_translation = np.array(data[key]['pose_translation'])

        n = pcd_sph.shape[0]
        pose_position_array = np.tile(pose_translation, (n,1))
        pcd_with_pose_position = np.hstack((pose_position_array, ))
        output = np.vstack((output, pcd_with_pose_position))
        if iter % 50 == 0:
            message = f"Preparing data ... ({iter}/{total_iter})"
            printProgress(message)
        iter += 1
    return output[1:,:]



if __name__ == "__main__":
    name = r'building_with_rotation'
    path = r'datasets/registered/' + name + r'.json'
    data = loadDataFromRegisteredSlam(path)
    training_data = preProcess(data)
    # output_name = name + r'.npy'
    # np.save(output_name, training_data)
    # print(f"file have been saved to {output_name}")


    


    
