import json
import numpy as np
import os
from einops import rearrange
from utility import printProgress
from numpy import sqrt, arctan2, array
import open3d as o3d

""" Preprocess data including:
1. Filter out points further than 5m (depend on datasets)
2. Find regsiters between frames
3. Obtain global coordinate (frame zero's coordinate) origins for each frame
4. Obtain view direction in global frame for each frame
5. Combine and save as one numpy array
6. Scale point cloud (including camera position) such that it fits into cube with edge length 1 
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


def prepareTrainingData(data):
    """Transform registered point clouds into giant array for model training
    input: data of type dict, with keys = frames
    output: n*6 numpy array
    """
    keys = list(data.keys())
    output = np.zeros((1,6))
    scene_points = np.zeros((1,3))
    iter = 0
    total_iter = len(keys)
    for key in keys:
        pcd_cart = np.array(data[key]['point_cloud'])
        pose_position = np.array(data[key]['pose'])
        pcd_sph = cart2sph(pcd_cart, pose_position)
        scene_points = np.vstack((scene_points, pcd_cart))
        scene_points = np.vstack((scene_points,pose_position))
        n = pcd_sph.shape[0]
        pose_position_array = np.tile(pose_position, (n,1))
        pcd_with_pose_position = np.hstack((pcd_sph, pose_position_array))
        output = np.vstack((output, pcd_with_pose_position))
        if iter % 50 == 0:
            message = f"Preparing data ... ({iter}/{total_iter})"
            printProgress(message)
        iter += 1

    # scale furthest distance between two points (including camera position) to 1        
    scene_points = scene_points[1:,:]   # discard first row
    max_xyz = np.max(scene_points, axis=0)
    min_xyz = np.min(scene_points, axis=0)
    max_distance = np.max(max_xyz - min_xyz)
    output[:,0] /= max_distance
    output[:,3:6] /= max_distance
    print(f"\nScaling factor is: {1 / max_distance}")

    return output[1:,:]



if __name__ == "__main__":
    name = r'box_plant2'
    input_path = r'datasets/registered/' + name + r'.json'
    data = loadDataFromRegisteredSlam(input_path)
    training_data = prepareTrainingData(data)
    output_name = name + r'.npy'
    np.save(output_name, training_data)
    print(f"file have been saved to {output_name}")
    


    
