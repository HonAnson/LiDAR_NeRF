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


if __name__ == '__main__':
    ### Choose which chuk and frame to be loaded here ###
    section = 9
    frame = 968
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
    
    # convert to np array and write to csv for livox data viewer
    points = np.array(data[str(frame)])
   
    ### Save to csv for visualization
    df_temp = pd.read_csv('local/visualize/dummy.csv')
    # pos_np = pos.numpy()
    pos_np = points
    df_temp = df_temp.head(pos_np.shape[0])
    df_temp['X'] = pos_np[:,0]
    df_temp['Y'] = pos_np[:,1]
    df_temp['Z'] = pos_np[:,2]
    df_temp.to_csv('local/visualize/visualize.csv', index=False)
    print("Data written to visualize.csv for visualization")



    

    
