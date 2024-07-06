import numpy as np
import open3d as o3d
import pandas as pd
import copy
import os
from preprocess import listFiles
### NOTE: target = transform(source)


def listFiles(directory):
    """Return list of files names in directory"""
    try:
        files = [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]
        return files
    except FileNotFoundError:
        print(f"Error: The directory '{directory}' does not exist.")
        return []
    except PermissionError:
        print(f"Error: You do not have permission to access the directory '{directory}'.")
        return []


def loadData(path):
    """ livox csv file from path to o3d point cloud"""
    df = pd.read_csv(path)
    points_xyz = df.to_numpy()
    points_xyz = points_xyz[:,8:11]
    point_cloud_output = o3d.geometry.PointCloud()
    point_cloud_output.points = o3d.utility.Vector3dVector(points_xyz)
    return point_cloud_output



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


def registerPair(path1, path2, draw_result = False):
    source = loadData(path1)
    target = loadData(path2)
    
 
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



def pairwiseRegistration(source, target):
    print("Apply point-to-plane ICP")
    max_correspondence_distance_coarse = 0.3
    max_correspondence_distance_fine = 0.05

    icp_coarse = o3d.pipelines.registration.registration_icp(
        source, target, max_correspondence_distance_coarse, np.identity(4),
        o3d.pipelines.registration.TransformationEstimationPointToPoint())
    icp_fine = o3d.pipelines.registration.registration_icp(
        source, target, max_correspondence_distance_fine,
        icp_coarse.transformation,
        o3d.pipelines.registration.TransformationEstimationPointToPoint())
    transformation_icp = icp_fine.transformation
    information_icp = o3d.pipelines.registration.get_information_matrix_from_point_clouds(
        source, target, max_correspondence_distance_fine,
        icp_fine.transformation)
    return transformation_icp, information_icp


def fullRegistration(pcds, max_correspondence_distance_coarse = 0.3,
                      max_correspondence_distance_fine = 0.05):
    pose_graph = o3d.pipelines.registration.PoseGraph()
    odometry = np.identity(4)
    pose_graph.nodes.append(o3d.pipelines.registration.PoseGraphNode(odometry))
    n_pcds = len(pcds)
    for source_id in range(n_pcds):
        for target_id in range(source_id + 1, n_pcds):
            transformation_icp, information_icp = pairwiseRegistration(
                pcds[source_id], pcds[target_id])
            print("Build o3d.pipelines.registration.PoseGraph")
            if target_id == source_id + 1:  # odometry case
                odometry = np.dot(transformation_icp, odometry)
                pose_graph.nodes.append(
                    o3d.pipelines.registration.PoseGraphNode(
                        np.linalg.inv(odometry)))
                pose_graph.edges.append(
                    o3d.pipelines.registration.PoseGraphEdge(source_id,
                                                             target_id,
                                                             transformation_icp,
                                                             information_icp,
                                                             uncertain=False))
            else:  # loop closure case
                pose_graph.edges.append(
                    o3d.pipelines.registration.PoseGraphEdge(source_id,
                                                             target_id,
                                                             transformation_icp,
                                                             information_icp,
                                                             uncertain=True))
    return pose_graph




if __name__ == "__main__":
    ### Choose directory here
    directory = r'datasets/csv/box_plant1_manual/'
    files = listFiles(directory)
    pcds = []
    for file in files:
        path = directory + file
        pcds.append(loadData(path))
    
    poseGraph = fullRegistration(pcds)
    
    
    ### For handling manual point clouds
    # path1 = r'datasets/csv/box_plant1_manual/box_plant1_manual_frame0.csv'
    # path2 = r'datasets/csv/box_plant1_manual/box_plant1_manual_frame5.csv'
    
    # source = loadData(path1)
    # target = loadData(path2)

    # transformation_icp, information_icp = pairwise_registration(source, target)
    # drawRegistrationResult(source, target, transformation_icp)
   

    

