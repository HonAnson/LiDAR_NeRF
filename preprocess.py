import json
import numpy as np
import os
import open3d as o3d

""" Preprocess data including:
1. Filter out points further than 5m (depend on datasets)
2. Find regsiters between frames
3. Obtain global coordinate (frame zero's coordinate) origins for each frame
4. Obtain view direction in global frame for each frame
5. Combine and save as one numpy array
"""

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
    

def shortPassFilter(points, threshold = 10):
    """ Filter points that are further than a distance away"""
    distance = (points[:,0]**2 + points[:,1]**2 + points[:,2]**2)**0.5
    mask = distance < threshold
    return points[mask]


def preProcess(name, threshold = 10):
    
    valid_names = ["box_plant1", "box_plant2", "building", "round_plant1", "round_plant2"]
    if name not in valid_names:
        print("Name not valid!")
        return
    
    # prepare path
    directory = r'datasets/json/' + name + r'/'
    files = listFiles(directory)
    files.sort()

    # prepare frame 0 as reference
    path = directory + files[0]
    with open(path, 'r') as file:
        data = json.load(file)
    pts_frame_0 = np.array(data[str(0)])
    pts_frame_0 = shortPassFilter(pts_frame_0, threshold)
       
    
    for file_name in files:
        # load file one by one
        path = directory + file_name
        with open(path, 'r') as file:
            data = json.load(file)
        
        for key in data:
            pts = data[key]
        
        # extract frame 0 pointcloud for reference


    
    return



if __name__ == "__main__":
    preProcess('box_plant1', threshold = 5)