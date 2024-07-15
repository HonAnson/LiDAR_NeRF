import json
import numpy as np
import os
from einops import rearrange
# import open3d as o3d

""" Preprocess data including:
1. Filter out points further than 5m (depend on datasets)
2. Find regsiters between frames
3. Obtain global coordinate (frame zero's coordinate) origins for each frame
4. Obtain view direction in global frame for each frame
5. Combine and save as one numpy array
"""


def loadDataFromRegisteredSlam(path):
    # load registered file
    with open(path,'r') as file:
        data = json.load(file)
    return data


def cart2sph(pcd_array, pose_position):
    pcd_local_aligned = pcd_array - pose_position
    x = pcd_local_aligned[:,0]
    y = pcd_local_aligned[:,1]
    z = pcd_local_aligned[:,2]
    XsqPlusYsq = x**2 + y**2
    r = np.sqrt(list(XsqPlusYsq + z**2))
    elev = np.arctan2(list(z), np.sqrt(list(XsqPlusYsq)))
    pan = np.arctan2(list(x), list(y))
    output = np.array([r, elev, pan])
    return rearrange(output, 'a b -> b a') #take transpose


def preProcess(data):
    keys = list(data.keys())
    output = np.zeros((1,6))
    iter = 0
    total_iter = len(keys)
    for key in keys:
        pcd_cart = np.array(data[key]['point_cloud'])
        pose_position = np.array(data[key]['pose'])
        pcd_sph = cart2sph(pcd_cart, pose_position)

        n = pcd_sph.shape[0]
        pose_position_array = np.tile(pose_position, (n,1))
        pcd_with_pose_position = np.hstack((pcd_sph, pose_position_array))
        output = np.vstack((output, pcd_with_pose_position))
        if iter % 50 == 0:
            print(f"Preparing data .. ({iter}/{total_iter})")
        iter += 1
    return output[1:,:]



if __name__ == "__main__":
    name = r'round_plant2'
    path = r'datasets/registered/' + name + r'.json'
    data = loadDataFromRegisteredSlam(path)
    training_data = preProcess(data)
    output_name = name + r'.npy'
    np.save(output_name, training_data)
    print(f"file have been saved to {output_name}")


    


    
