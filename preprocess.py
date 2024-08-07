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


def getDistanceAndDirection(pcd_array, pose_position):
    """ Convert point cloud and camera position to training data 
    Input: 
    pcd_array: n*3 numpy array
    pose_position: 1*3 numpy array

    Output:
    distance: n numpy array
    ray_direction n*3 numpy array, where each row represent a unit vector
    """
    pcd_local_aligned = pcd_array - pose_position
    x, y, z = pcd_local_aligned[:,0], pcd_local_aligned[:,1], pcd_local_aligned[:,2]
    distance = sqrt(x**2 + y**2 + z**2)
    distance =  rearrange(distance, 'n -> n 1')
    ray_direction = pcd_local_aligned / distance

    return distance, ray_direction


def prepareTrainingData(data):
    """Transform registered point clouds into giant array for model training
    input: data of type dict, with keys = frames
    output: n*6 numpy array
    """
    keys = list(data.keys())
    output = np.zeros((1,7))
    scene_points = np.zeros((1,3)) # for scaling point cloud
    iter = 0
    total_iter = len(keys)
    for key in keys:
        pcd_cart = np.array(data[key]['point_cloud'])
        ray_origin = np.array(data[key]['pose'])
        scene_points = np.vstack((scene_points, pcd_cart, ray_origin))     # for scaling the scene
        distance, ray_direction = getDistanceAndDirection(pcd_cart, ray_origin)

        # Stack distance, ray origin, ray direction into output array
        n = distance.shape[0]
        ray_origin_tiled = np.tile(ray_origin, (n,1))
        training_data = np.hstack((ray_origin_tiled,ray_direction, distance))
        output = np.vstack((output, training_data))

        # Print load message
        if iter % 50 == 0:
            message = f"Preparing data ... ({iter}/{total_iter})"
            printProgress(message)
        iter += 1

    # scale furthest distance between two points (including camera position) to 2 and centred at 0        
    scene_points = scene_points[1:,:]   # discard first row
    max_xyz = np.max(scene_points, axis=0)
    min_xyz = np.min(scene_points, axis=0)
    max_distance = np.max(max_xyz - min_xyz)
    scene_centre = np.mean(scene_points, axis=0)
    output[:,0:3] -= scene_centre
    output[:,0:3] /= (max_distance*0.5)       # normalize camera position
    output[:,-1] /= (max_distance*0.5)        # normalize depth
    print(f"\nScaling factor is: {1 / max_distance}")
    return output[1:,:]


if __name__ == "__main__":
    name = r'building'
    input_path = r'datasets/registered/' + name + r'.json'
    data = loadDataFromRegisteredSlam(input_path)
    training_data = prepareTrainingData(data)
    output_name = name + r'.npy'
    # np.save(output_name, training_data)
    # print(f"file have been saved to {output_name}")
    


    
