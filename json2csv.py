import json
import os
import numpy as np
import pandas as pd


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

def json2csv(data, frame, out_path):
    """ Read point cloud from a frame in json and stor in a livox csv"""
    # convert to np array and write to csv for livox data viewer
    pts_np = np.array(data[str(frame)])
   
    # Load dummy csv
    df_temp = pd.read_csv('local/visualize/dummy.csv')
    df_temp = df_temp.head(pts_np.shape[0])

    # write to dummy csv
    df_temp['X'] = pts_np[:,0]
    df_temp['Y'] = pts_np[:,1]
    df_temp['Z'] = pts_np[:,2]
    df_temp.to_csv('local/visualize/visualize.csv', index=False)
    print("Data written to ", out_path)
    return 


if __name__ == '__main__':
    ### Choose which chuk and frame to be loaded here ###
    section = 0
    frame = 1
    directory = r'datasets/json/box_plant1/'
    ##########
    files = listFiles(directory)
    files.sort()
    path = directory + files[section]
    
    # load data from path
    with open(path, 'r') as file:
        data = json.load(file)
    num_keys = len(data.keys())
    print(f"Data loaded, number of frames = {num_keys}")

    
    for key in data:
        out_directory = r'datasets/csv/box_plant1/'
        out_filename = r'box_plant1_frame' + key + r'.csv'
        out_path = out_directory + out_filename
        json2csv(data, int(key), out_path)
        

    

    
