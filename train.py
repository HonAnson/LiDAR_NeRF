import numpy as np
import pandas as pd
import math as m
from einops import rearrange
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import torch.optim as optim
import math


# Load data from dataset directory, filterout irralevent datas
def loadData():
    # Specify the directory path
    dataset_path = 'datasets/testing1'
    # List all files in the specified path, ignoring directories
    files = [f for f in os.listdir(dataset_path) if os.path.isfile(os.path.join(dataset_path, f))]
    files.sort()
    # read the files
    points_xyz = []
    for s in files:
        path = dataset_path + s
        df = pd.read_csv(path)
        a = df.to_numpy()
        b = a[:,8:11]
        points_xyz.append(b)
    return points_xyz   # type = list


### Process the data
def prepareData(points_xyz):
    # (1) Convert to spherical coorindate
    # NOTE: points in spherical coordinate are arranged: [r, elev, pan]
    points_sphere = []
    for points in points_xyz:
        points_sphere.append(cart2sph(points))

    #### HARD CODED HERE ####
    # Translation vectors for points in each view, we are using camera centre at first frame as origin of world coordinate
    # NOTE: translation vectors below are found by assuming transformation between frames are translations, and obatined by manually finding corrspondance
    # They are translation of the same corrspondance across different frames
    t0 = np.array([0,0,0])
    t1 = np.array([-0.671,-0.016,0.215])
    t2 = np.array([-1.825,-0.091,0.147])
    t3 = np.array([-2.661,-0.263,0.166])
    t4 = np.array([-3.607,-0.156,0.039])
    translations = [t0, t1, t2, t3, t4]
    return dataset   # type = numpyArray




class LiDAR_NeRF(nn.Module):
    def __init__(self, embedding_dim_pos = 10, embedding_dim_dir = 4, hidden_dim = 256):
        super(LiDAR_NeRF, self).__init__()
        self.embedding_dim_dir = embedding_dim_dir
        self.embedding_dim_pos = embedding_dim_pos
        self.layers = nn.Sequential(
            nn.Linear(embedding_dim_pos * 3 + embedding_dim_dir*2, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),               
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),               
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),               
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),               
            nn.Linear(512, 1)          # Output layer with 1 output
        )
        
    @staticmethod
    def positional_encoding(x, L):
        out = [x]
        for j in range(L):
            out.append(torch.sin(2 ** j * x))
            out.append(torch.cos(2 ** j * x))
        return torch.cat(out, dim=1)

    def forward(self, o, d):
        emb_x = self.positional_encoding(o, self.embedding_dim_pos)
        emb_d = self.positional_encoding(d, self.embedding_dim_dir)
        
        return self.layers(x)


    def lossBCE(self, r, y, sampled_pos):  # r = distance from lidar measurement, y = function output at different points, it is a vector
        y_sigmoid_value = nn.sigmoid(y)
        r_sigmoid_value = nn.sigmoid(-sampled_pos-r)
        loss = nn.CrossEntropyLoss(r_sigmoid_value, y_sigmoid_value)
        return loss









### Train the model
def trainModel(dataset):
        
    return


#################################################
# Utility functions
# convert pointcloud from cartisean coordinate to spherical coordinate
def cart2sph(xyz):
    x = xyz[:,0]
    y = xyz[:,1]
    z = xyz[:,2]
    XsqPlusYsq = x**2 + y**2
    r = np.sqrt(list(XsqPlusYsq + z**2))
    elev = np.arctan2(list(z), np.sqrt(list(XsqPlusYsq)))
    pan = np.arctan2(list(x), list(y))
    output = np.array([r, elev, pan])
    return rearrange(output, 'a b -> b a') #take transpose
#######################################







if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using {device} device")
    points = loadData()




