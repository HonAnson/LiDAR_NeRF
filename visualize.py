import open3d
import torch
from train import LiDAR_NeRF
from einops import rearrange
from numpy import cos, sin, array, sqrt, arctan2
from utility import printProgress
import numpy as np
import mcubes
import trimesh


def renderModel(model, num_bins, boundary):
    t = np.linspace(-boundary, boundary, num_bins)    
    query_pts = np.stack(np.meshgrid(t, t, t), -1).astype(np.float32)
    shape = query_pts.shape
    flat = rearrange(query_pts, 'x y z d -> (x y z) d')

    # Query the model by chunk to get the density
    chunk = 1024*64
    num_points = flat.shape[0]
    raw = []
    for i in range(0, num_points, chunk):
        message = f"Rendering model... ({int(i/chunk)}/{int(num_points / chunk)})"
        printProgress(message)
        query = torch.tensor(flat[i:i+chunk, :])
        with torch.no_grad():
            raw.append((model(query)).detach().numpy())
    raw = np.concatenate(raw)
    raw = rearrange(raw, '(X Y Z) 1 -> X Y Z 1', X = num_bins, Y = num_bins, Z = num_bins)
    sigma = np.maximum(raw[...,-1], 0.)
    # Deploy marching cube algorithm
    threshold = 10
    print('\nFraction occupied', np.mean(sigma > threshold))
    vertices, triangles = mcubes.marching_cubes(sigma, threshold)
    print('Done', vertices.shape, triangles.shape)
    mesh = trimesh.Trimesh(vertices / num_bins - .5, triangles)
    mesh.show()
    return



if __name__ == "__main__":
    # NOTE: camera points at [1,0,0] when unrotated
    weights_path = r'local/models/ver_cumulative_trial20.pth'
    output_path = r'local/visualize/visualize.csv'

    ### Parameters for the model
    HIDDEN_DIM = 512
    EMBEDDING_DIM_DIR = 10
    EMBEDDING_DIM_POS = 15

    ### Rendering Parameters
    NUM_BINS = 128
    BOUNDARIES = 0.5

    ###########################
    model_evel = LiDAR_NeRF(embedding_dim_pos=EMBEDDING_DIM_POS, 
                       embedding_dim_dir=EMBEDDING_DIM_DIR, 
                       hidden_dim=HIDDEN_DIM)
    
    #### Load the model and try to "visualize" the model's datapoints
    model_evel.load_state_dict(torch.load(weights_path))
    model_evel.eval(); # Set the model to inference mode
    renderModel(model_evel, NUM_BINS, BOUNDARIES)

    



