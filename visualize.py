import open3d
import numpy 
import torch
from train import LiDAR_NeRF
from einops import repeat, rearrange
from numpy import cos, sin, array, sqrt, arctan2
import pandas as pd
from utility import printProgress
import numpy as np
import os, sys
# os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import numpy as np
# import imageio
import matplotlib.pyplot as plt
import mcubes
import trimesh



def sph2cart(ang):
    ele = ang[:,0]
    pan = ang[:,1]
    x = cos(ele)*cos(pan)
    y = cos(ele)*sin(pan)
    z = sin(ele)
    output = array([x,y,z])
    return rearrange(output, 'a b -> b a') #take transpose

def getAng(points):
    """ Convert a n*3 point cloud, 
    with camera centre at pose_position from cartesian coordinate in global frame
    to global algined spherical coordinate at camera frame
    
    """
    x, y, z = points[0], points[1], points[2]
    XsqPlusYsq = x**2 + y**2
    elev = arctan2(z, sqrt(XsqPlusYsq))
    pan = arctan2(y, x)
    return elev, pan

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
    print(f'\nVisualizing output saved to {output_path}')
    return

def visualizeDir(model_path, output_path, position, direction):
    """ Visualize reconstruction from model and position"""
    #### Load the model and try to "visualize" the model's datapoints
    model_evel = LiDAR_NeRF(hidden_dim=512, embedding_dim_dir=15, device = 'cpu')
    model_evel.load_state_dict(torch.load(model_path))
    model_evel.eval(); # Set the model to inference mode
    
    ### evaluate direction
    ele_dir, pan_dir = getAng(direction)
    ele_dir = float(ele_dir)
    pan_dir = float(pan_dir)
    position_tiled = repeat(position, 'd-> r d', r = 100000)

    ### Render some structured pointcloud for evaluation
    with torch.no_grad():
        dist = 0.05 # initial distanc for visualization
        pos = torch.tensor(position_tiled, dtype = torch.float32)
        ele = torch.linspace(-0.1, 0.1, 100)
        pan = torch.linspace(-0.2, 0.2, 1000)

        ele_tiled = repeat(ele, 'n -> (r n) 1', r = 1000)
        pan_tiled = repeat(pan, 'n -> (n r) 1', r = 100)
        ang = torch.cat((ele_tiled, pan_tiled), dim=1)

        # direction for each "point" from camera centre
        directions = torch.tensor(sph2cart(array(ang)))
        iterations = 1000
        for i in range(iterations):
            output2 = model_evel(pos, ang)
            temp = torch.sign(output2)
            pos += directions * dist * temp
            printProgress(f'Visualizing... ({i}/{iterations})')

    ### Save to csv for visualization
    df_temp = pd.read_csv('local/visualize/dummy.csv')
    df_temp = df_temp.head(100000)
    pos_np = pos.numpy()
    df_temp['X'] = pos_np[:,0]
    df_temp['Y'] = pos_np[:,1]
    df_temp['Z'] = pos_np[:,2]
    df_temp.to_csv(output_path, index=False)
    print(f'\nVisualizing output saved to {output_path}')

    return



def renderModel(model):
    num_bins = 64
    t = np.linspace(-1.2, 1.2, num_bins)
    query_pts = np.stack(np.meshgrid(t, t, t), -1).astype(np.float32)
    shape = query_pts.shape
    flat = rearrange(query_pts, 'x y z d -> (x y z) d')

    # Query the model by chunk to get the density
    chunk = 1024*64
    num_points = flat.shape[0]
    raw = []
    count = 0
    for i in range(0, num_points, chunk):
        print(count)
        query = torch.tensor(flat[i:i+chunk, :])
        with torch.no_grad():
            raw.append((model(query)).detach().numpy())
        count += 1
    raw = np.concatenate(raw)
    raw = rearrange(raw, '(X Y Z) 1 -> X Y Z 1', X = num_bins, Y = num_bins, Z = num_bins)


    # # fn = lambda i0, i1 : net_fn(flat[i0:i1,None,:], viewdirs=np.zeros_like(flat[i0:i1]), network_fn=render_kwargs_test['network_fine'])

    # chunk = 1024*64
    # raw = np.concatenate([fn(i, i+chunk).numpy() for i in range(0, flat.shape[0], chunk)], 0)
    # raw = np.reshape(raw, list(shape[:-1]) + [-1])
    sigma = np.maximum(raw[...,-1], 0.)


    # Deploy marching cube algorithm
    threshold = 2
    print('fraction occupied', np.mean(sigma > threshold))
    vertices, triangles = mcubes.marching_cubes(sigma, threshold)
    print('done', vertices.shape, triangles.shape)


    mesh = trimesh.Trimesh(vertices / num_bins - .5, triangles)
    mesh.show()

    return



if __name__ == "__main__":
    # NOTE: camera points at [1,0,0] when unrotated
    model_path = r'local/models/ver_cumulative_trial00.pth'
    output_path = r'local/visualize/visualize.csv'


    ### Parameters for the model
    HIDDEN_DIM = 512
    EMBEDDING_DIM_DIR = 10
    EMBEDDING_DIM_POS = 15

    ###########################
    model_evel = LiDAR_NeRF(embedding_dim_pos=EMBEDDING_DIM_POS, 
                       embedding_dim_dir=EMBEDDING_DIM_DIR, 
                       hidden_dim=HIDDEN_DIM)
    
    #### Load the model and try to "visualize" the model's datapoints
    model_evel.load_state_dict(torch.load(model_path))
    model_evel.eval(); # Set the model to inference mode
    renderModel(model_evel)

    
    # visualize360(model_path,output_path)
    # position = array([0,0,0])
    # direction = array([1,0,0])
    # visualizeDir(model_path, output_path, position, direction)



