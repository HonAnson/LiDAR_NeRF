import open3d
import numpy 
import torch
from train import LiDAR_NeRF
from einops import repeat, rearrange
from numpy import cos, sin, array, sqrt, arctan2
import pandas as pd
from utility import printProgress
import numpy as np

def sph2cart(ang):
    ele = ang[:,0]
    pan = ang[:,1]
    x = cos(ele)*cos(pan)
    y = cos(ele)*sin(pan)
    z = sin(ele)
    output = array([x,y,z])
    return rearrange(output, 'a b -> b a') #take transpose

def cart2sph(pcd_array, pose_position):
    """ Convert a n*3 point cloud, 
    with camera centre at pose_position from cartesian coordinate in global frame
    to global algined spherical coordinate at camera frame
    
    """
    pcd_local_aligned = pcd_array - pose_position
    x, y, z = pcd_local_aligned[:,0], pcd_local_aligned[:,1], pcd_local_aligned[:,2]
    XsqPlusYsq = x**2 + y**2
    r = sqrt(XsqPlusYsq + z**2)
    elev = arctan2(z, sqrt(XsqPlusYsq))
    pan = arctan2(y, x)
    output = array([r, elev, pan])
    return rearrange(output, 'a b -> b a') #take transpose



def visualize360(model_path, output_path):
    """ Visualize reconstruction from model and position"""
    #### Load the model and try to "visualize" the model's datapoints
    model_evel = LiDAR_NeRF(hidden_dim=512, embedding_dim_dir=15, device = 'cpu')
    model_evel.load_state_dict(torch.load(model_path))
    model_evel.eval(); # Set the model to inference mode

    ### Render some structured pointcloud for evaluation
    with torch.no_grad():
        dist = 0.05 # initial distanc for visualization
        pos = torch.zeros((100000,3))
        ele = torch.linspace(-0.34, 0.3, 100)
        pan = torch.linspace(-3.14, 3.14, 1000)
        ele_tiled = repeat(ele, 'n -> (r n) 1', r = 1000)
        pan_tiled = repeat(pan, 'n -> (n r) 1', r = 100)
        ang = torch.cat((ele_tiled, pan_tiled), dim=1)

        # direction for each "point" from camera centre
        directions = torch.tensor(sph2cart(array(ang)))

        for i in range(1000):
            output2 = model_evel(pos, ang)
            temp = torch.sign(output2)
            pos += directions * dist * temp
            printProgress(f'visualizing... ({i}/100)')

    ### Save to csv for visualization
    df_temp = pd.read_csv('local/visualize/dummy.csv')
    df_temp = df_temp.head(100000)
    pos_np = pos.numpy()
    df_temp['X'] = pos_np[:,0]
    df_temp['Y'] = pos_np[:,1]
    df_temp['Z'] = pos_np[:,2]
    df_temp.to_csv(output_path, index=False)
    print(f'Visualizing output saved to {output_path}')
    return




def visualizeDir(model_path, output_path, direction):
    """ Visualize reconstruction from model and position"""
    #### Load the model and try to "visualize" the model's datapoints
    model_evel = LiDAR_NeRF(hidden_dim=512, embedding_dim_dir=15, device = 'cpu')
    model_evel.load_state_dict(torch.load(model_path))
    model_evel.eval(); # Set the model to inference mode
    
    ### evaluate direction

    ### Render some structured pointcloud for evaluation
    with torch.no_grad():
        dist = 0.05 # initial distanc for visualization
        pos = torch.zeros((100000,3))
        ele = torch.linspace(-0.34, 0.3, 100)
        pan = torch.linspace(-3.14, 3.14, 1000)
        ele_tiled = repeat(ele, 'n -> (r n) 1', r = 1000)
        pan_tiled = repeat(pan, 'n -> (n r) 1', r = 100)
        ang = torch.cat((ele_tiled, pan_tiled), dim=1)

        # direction for each "point" from camera centre
        directions = torch.tensor(sph2cart(array(ang)))

        for i in range(1000):
            output2 = model_evel(pos, ang)
            temp = torch.sign(output2)
            pos += directions * dist * temp
            printProgress(f'visualizing... ({i}/100)')

    ### Save to csv for visualization
    df_temp = pd.read_csv('local/visualize/dummy.csv')
    df_temp = df_temp.head(100000)
    pos_np = pos.numpy()
    df_temp['X'] = pos_np[:,0]
    df_temp['Y'] = pos_np[:,1]
    df_temp['Z'] = pos_np[:,2]
    df_temp.to_csv(output_path, index=False)
    print(f'Visualizing output saved to {output_path}')

    return

if __name__ == "__main__":
    model_path = r'local/models/version4_trial2.pth'
    output_path = r'local/visualize/visualize.csv'
    visualize360(model_path,output_path)







# def getUnitVectorfromImage(direction, focal_length, height = 1, width = 1, width_resolution = 1000, height_resolution = 1000):
    
#     # Initialize the output array
#     unit_vectors = np.zeros((1000000, 3))
    
#     # Compute the pixel coordinates
#     x_coords, y_coords = np.meshgrid(np.arange(height_resolution), np.arange(width_resolution))
#     x_coords = x_coords * (width / width_resolution)
#     y_coords = y_coords * (height / height_resolution)
    
#     # Flatten the coordinates
#     x_coords = x_coords.flatten()
#     y_coords = y_coords.flatten()

#     # Compute the direction vectors
#     for i in range(height_resolution*width_resolution):
#         x = (x_coords[i] - width / 2)
#         y = (y_coords[i] - height / 2)
#         z = focal_length
#         vec = np.array([x, y, z])
#         unit_vectors[i] = vec / np.linalg.norm(vec)
#     return unit_vectors

