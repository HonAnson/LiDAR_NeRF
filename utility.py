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
    """ Visualize point cloud in format n*3 numpy array"""
    # Convert the numpy array to an Open3D PointCloud object
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)

    # Visualize the point cloud
    o3d.visualization.draw_geometries([pcd])
    return

if __name__ == "__main__":
    pass
