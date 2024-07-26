import os
import sys
import time
import open3d as o3d


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

def printProgress(message):
    toprint = "\r" + message
    print(toprint, end = "")
    return


def quickVizNumpy(points):
    # Convert the numpy array to an Open3D PointCloud object
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)

    # Visualize the point cloud
    o3d.visualization.draw_geometries([pcd])
    return

def quickVizTwoNumpy(points1, points2):
    """
    Visualize two point clouds with different colors using Open3D.
    Parameters:
    - points1 (numpy.ndarray): The first n*3 numpy array of points.
    - points2 (numpy.ndarray): The second n*3 numpy array of points.
    """
    # Convert numpy arrays to Open3D point clouds
    pcd1 = o3d.geometry.PointCloud()
    pcd1.points = o3d.utility.Vector3dVector(points1)
    pcd1.paint_uniform_color([1, 0, 0])  # Paint red
    
    pcd2 = o3d.geometry.PointCloud()
    pcd2.points = o3d.utility.Vector3dVector(points2)
    pcd2.paint_uniform_color([0, 0, 1])  # Paint blue
    
    # Visualize the point clouds
    o3d.visualization.draw_geometries([pcd1, pcd2])


if __name__ == "__main__":
    pass
