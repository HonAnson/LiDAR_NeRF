import numpy as np
import matplotlib.pyplot as plt
from einops import rearrange
import rosbags
import json
import open3d as o3d
import pandas as pd
import os
import copy
from preprocess import listFiles



def shortPassFilter(points, threshold = 10):
    """ Filter points that are further than a distance away"""
    distance = (points[:,0]**2 + points[:,1]**2 + points[:,2]**2)**0.5
    mask = distance < threshold
    return points[mask]


def longtPassFilter(points, threshold = 0.1):
    """ Filter points that are closer than a distance"""
    distance = (points[:,0]**2 + points[:,1]**2 + points[:,2]**2)**0.5
    mask = distance > threshold
    return points[mask]

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
    frame_idx = 0
    # load poses
    with open(path_pose,'r') as file:
        poses = json.load(file)
    pose_keys = list(poses.keys())
    pose_keys.sort()

    for path_pcd in paths_pcd:
        with open(path_pcd,'r') as file:
            pcds = json.load(file)

        for pcd_key in pcds:
            print(frame_idx)
            # skip the first 5 frames, as they are to be discarded
            if frame_idx < 5:
                frame_idx += 1
                continue
            try:
                pcd = np.array(pcds[pcd_key])
                pose_key = pose_keys[frame_idx - 5]
                translation = np.array(poses[pose_key]['translation'])
                rotation = np.array(poses[pose_key]['rotation'])

                pcd = shortPassFilter(pcd, 20)
                pcd = longtPassFilter(pcd)
                pcd = transformPointCloud(pcd, translation, rotation)
                output[frame_idx] = {'point_cloud':pcd.tolist(), 'pose':translation.tolist()}
                frame_idx += 1

            except:
                frame_idx += 1
    return output



if __name__ == "__main__":
    name = r'building'
    directory_pdc = r'datasets/json/' + name + r'/'
    files_pcd = listFiles(directory_pdc)
    files_pcd.sort()
    paths_pcd = [directory_pdc + file_name for file_name in files_pcd]
    path_pose = r'datasets/pose/' + name + r'_pose.json'
    registered_pcd = registerFromSlam(paths_pcd, path_pose)
    with open("building.json", "w") as outfile: 
        json.dump(registered_pcd, outfile)








