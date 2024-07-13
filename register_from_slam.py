import numpy as np
import matplotlib.pyplot as plt
from einops import rearrange
import rosbags
import json
import open3d as o3d
import pandas as pd
import os
import copy
from preprocess import listFiles, shortPassFilter




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

def transformPointCloud(point_cloud, translation, quaternion):
    """
    Apply translation and rotation to a point cloud.
    
    Args:
        point_cloud (np.ndarray): An n x 3 array representing the point cloud.
        translation (np.ndarray): A 1 x 3 array representing the translation vector.
        quaternion (np.ndarray): A 1 x 4 array representing the quaternion for rotation.
    
    Returns:
        np.ndarray: The transformed point cloud.
    """
    # Validate inputs
    assert point_cloud.shape[1] == 3, "Point cloud must be an n x 3 array."
    assert translation.shape == (3,), "Translation must be a 1 x 3 array."
    assert quaternion.shape == (4,), "Quaternion must be a 1 x 4 array."

    R = quat2RotationMatrix(quaternion)
    rotated_pcd = R@(np.transpose(point_cloud))
    global_pcd = np.transpose(rotated_pcd) + translation
    return global_pcd


def registerFromSlam(paths_pcd, path_pose):
    output = {}
    frame_counter = 0
    # load poses
    with open(path_pose,'r') as file:
        poses = json.load(file)
    pose_keys = list(poses.keys())

    for path_pcd in paths_pcd:
        with open(path_pcd,'r') as file:
            pcds = json.load(file)

        for key_pcd in pcds:
            print(frame_counter)
            # skip the first 5 frames, as they are to be discarded
            if frame_counter < 5:
                frame_counter += 1
                continue
            try:
                key_pose = pose_keys[frame_counter - 5]
                pcd = np.array(pcds[key_pcd])
                translation = np.array(poses[key_pose]['translation'])
                rotation = np.array(poses[key_pose]['rotation'])

                pcd = shortPassFilter(pcd)
                pcd = transformPointCloud(pcd, translation, rotation)

                output[frame_counter] = pcd.tolist()

                frame_counter += 1
            except:
                frame_counter += 1
    return output



if __name__ == "__main__":
    name = r'box_plant2'
    directory_pdc = r'datasets/json/' + name + r'/'
    files_pcd = listFiles(directory_pdc)
    files_pcd.sort()
    paths_pcd = [directory_pdc + file_name for file_name in files_pcd]
    path_pose = r'datasets/pose/' + name + r'_pose.json'

    registered_pcd = registerFromSlam(paths_pcd, path_pose)
    with open("testing.json", "w") as outfile: 
        json.dump(registered_pcd, outfile)




