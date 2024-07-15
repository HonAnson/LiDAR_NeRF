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
import json






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
        emb_input = torch.cat(emb_x, emb_d)
        return self.layers(emb_input)

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




