import numpy as np
import open3d as o3d
import pandas as pd
import copy
from preprocess import listFiles
### NOTE: target = transform(source)

def loadData(path):
    """ livox csv file to numpy array of point cloud"""
    df = pd.read_csv(path)
    points_xyz = df.to_numpy()
    return points_xyz[:,8:11]

### Function for global registration
def preprocessPointCloud(pcd, voxel_size):
    pcd_down = pcd.voxel_down_sample(voxel_size)

    radius_normal = voxel_size * 2
    pcd_down.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))

    radius_feature = voxel_size * 5
    pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd_down,
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))
    return pcd_down, pcd_fpfh


def globalRegistration(source_down, target_down, source_fpfh,
                                target_fpfh, voxel_size):
    distance_threshold = voxel_size * 1.5
    # print(":: RANSAC registration on downsampled point clouds.")
    # print("   Since the downsampling voxel size is %.3f," % voxel_size)
    # print("   we use a liberal distance threshold %.3f." % distance_threshold)
    result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        source_down, target_down, source_fpfh, target_fpfh, True,
        distance_threshold,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        3, [
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(
                0.9),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(
                distance_threshold)
        ], o3d.pipelines.registration.RANSACConvergenceCriteria(100000, 0.999))
    return result

def localRegistration(global_reg, source, target):
    trans_init = np.asarray(global_reg.transformation)
    threshold=0.02
    reg_p2p = o3d.pipelines.registration.registration_icp(
        source, target, threshold, trans_init,
        o3d.pipelines.registration.TransformationEstimationPointToPoint())
    
    return reg_p2p


def drawRegistrationResult(source, target, transformation):
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    source_temp.paint_uniform_color([1, 0.706, 0])
    target_temp.paint_uniform_color([0, 0.651, 0.929])
    source_temp.transform(transformation)
    o3d.visualization.draw_plotly([source_temp, target_temp])
    return

def register(path1, path2, draw_result = False):
    pts_source = loadData(path1)
    pts_target = loadData(path2)
    
    # Create Open3D PointCloud objects
    source = o3d.geometry.PointCloud()
    source.points = o3d.utility.Vector3dVector(pts_source)

    target= o3d.geometry.PointCloud()
    target.points = o3d.utility.Vector3dVector(pts_target)

    # Conduct downsampling and get point cloud fpfh
    voxel_size=0.05
    source_down, source_fpfh = preprocessPointCloud(source, voxel_size)
    target_down, target_fpfh = preprocessPointCloud(target, voxel_size)
    # register
    global_reg_result = globalRegistration(source_down, target_down, source_fpfh, target_fpfh, voxel_size)
    local_reg_result = localRegistration(global_reg_result, source_down, target_down)
    if draw_result is True:
        drawRegistrationResult(source_down, target_down, local_reg_result.transformation)
    return local_reg_result



if __name__ == "__main__":
    ### Choose directory here
    directory = r''
    ### For handling manual point clouds
    path1 = r'datasets/box_plant1_manual_frame0.csv'
    path2 = r'datasets/box_plant1_manual_frame60.csv'
    registration_result = register(path1, path2)
    

