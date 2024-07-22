import numpy as np
import pandas as pd
import math as m
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import torch.optim as optim
import json
# from tqdm import tqdm       # for showing progress when training
# import open3d as o3d        # for getting point cloud register
from einops import rearrange, repeat
from numpy import sin, cos
from utility import printProgress

def getDirections(angles):
    """ Convert torch tensor of angles to 
    cartiseian coordinate unit vector pointing that direction
    """
    elev, pan = angles[:,0], angles[:,1]
    x_tilde, y_tilde, z_tilde = cos(elev)*cos(pan), cos(elev)*sin(pan), sin(elev)      
    unit_vectors = torch.vstack([x_tilde, y_tilde, z_tilde])
    return unit_vectors





def getSpacing(num_points, num_bins):
    """return a [num_points*num_bins, 1] pytorch tensor
    
    """
    # TODO: add hyperparameter for tuning "slope" of inverse sigmoid function
    # create a list of magnitudes with even spacing from 0 to 1
    t = torch.linspace(0,1, num_bins).expand(num_points, num_bins)  # [batch_size, num_bins//2]
    
    # preterb the spacing
    mid = (t[:, :-1] + t[:, 1:]) / 2.
    lower = torch.cat((t[:, :1], mid), -1)
    upper = torch.cat((mid, t[:, -1:]), -1)
    u = torch.rand(t.shape)
    t = lower + (upper - lower) * u  # [batch_size, nb_bins//2]
    # hard code start and end value of spacing to avoid infinity
    t[:,0] = 1e-4
    t[:,-1] = 0.995
    # apply inverse sigmoid function to even spacing
    t = rearrange(t, 'a b -> (a b) 1')  # [num_bins*batch_size, 1] 
    t = torch.log(t / ((1 - t) + 1e-8)) 
    return t  

def getSamples(centres, angles, r, num_bins = 100):
    num_points = r.shape[0]
    elev = angles[:,0]
    pan = angles[:,1]
    x_tilde, y_tilde, z_tilde = cos(elev)*cos(pan), cos(elev)*sin(pan), sin(elev)      
    unit_vectors = torch.vstack([x_tilde, y_tilde, z_tilde])

    # process vectors: [3, num_points] -> [num_points*num_bins, 3]
    unit_vectors_repeated = repeat(unit_vectors, 'c n -> (n b) c', b = num_bins)
    r_repeated = repeat(r, 'n -> (n b) 1', b = num_bins)
    t = getSpacing(num_points, num_bins)
    sample_magnitudes = t + r_repeated
    pos = unit_vectors_repeated*sample_magnitudes      # [num_bins*num_points, 3]

    # tile the origin values
    # complete getting sample position by adding camera centre position to sampled position
    centres_tiled = torch.tensor(repeat(centres, 'n c -> (n b) c', b = num_bins)) # [num_bin*batch_size, 3]
    pos = centres_tiled + pos

    # tile the angle too
    angles_tiled = torch.tensor(repeat(angles, 'n c -> (n b) c', b = num_bins))  # [num_bin*batch_size, 2]
    return pos, angles_tiled, centres_tiled




# returns pytorch tensor of sigmoid of projected SDF
def getTargetValues(sample_positions, gt_distance, origins, num_bins=100):
    # calculate distance from sample_position
    temp = torch.tensor((sample_positions - origins)**2)
    pos_distance = torch.sqrt(torch.sum(temp, dim=1, keepdim=True))
    
    # find the "projected" value
    sigmoid = nn.Sigmoid()
    values = sigmoid(-(pos_distance - gt_distance))
    return values


class LiDAR_NeRF(nn.Module):
    def __init__(self, embedding_dim_pos = 10, embedding_dim_dir = 4, hidden_dim = 256, device = 'cuda'):
        super(LiDAR_NeRF, self).__init__()
        self.device = device
        self.embedding_dim_dir = embedding_dim_dir
        self.embedding_dim_pos = embedding_dim_pos
        self.block1 = nn.Sequential(
            nn.Linear(embedding_dim_pos * 6 + 3, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),               
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),               
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),               
        )
        
        self.block2 = nn.Sequential(
            nn.Linear(embedding_dim_pos * 6 + 3 + hidden_dim, hidden_dim), nn.ReLU(),               
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),               
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),               
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim,1), nn.ReLU()      # relu for last layer, as we are predicting density of at locations
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
        temp = self.block1(emb_x)
        # temp2 = torch.cat((temp, emb_x), dim=1).to(dtype=torch.float32) # add skip input
        temp2 = torch.cat((temp, emb_x), dim=1) # add skip input
        density = self.block2(temp2)
        return density


def train(model, optimizer, scheduler, dataloader, device = 'cuda', num_epoch = int(1e5), num_bins = 100):
    training_losses = []
    num_batch_in_data = len(dataloader)
    count = 0
    for epoch in range(num_epoch):
        for iter, batch in enumerate(dataloader):

            # parse the batch
            gt_dist = batch[:,0]
            angles = batch[:,1:3]
            centers = batch[:,3:6]

            sample_pos, sample_ang, sample_org = getSamples(centers, angles, gt_dist, num_bins=num_bins)

            # tile distances
            gt_dist_tiled = repeat(gt_dist, 'b -> (b n) 1', n=num_bins)

            # stack the upsampled position to sampled positions
            pos = sample_pos.to(device)
            ang = sample_ang.to(device)

            gt_dist = (torch.vstack((gt_dist_tiled,upsample_gt_dist))).to(device)
            org = (torch.vstack((sample_org, upsample_pos))).to(device)
            
            # inference
            rendered_value = model(pos, ang)


            actual_value_sigmoided = (getTargetValues(pos, gt_dist, org)).to(dtype = torch.float32)
            # loss = lossBCE(rendered_value, actual_value_sigmoided)  # + lossEikonal(model)

            # loss_bce = nn.BCELoss()
            loss_MSE = nn.MSELoss()
            loss = loss_MSE(rendered_value_sigmoid, actual_value_sigmoided)         # + lossEikonal
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            ### Prin loss messages
            if count % 100 == 0:
                training_losses.append(loss.item())
            count += 1
            message = f"Training model... epoch: ({epoch}/{num_epoch}) | iteration: ({iter}/{num_batch_in_data}) | loss: {loss.item()}"
            printProgress(message)

        scheduler.step()
    return training_losses



if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using {device} device")

    data_path = r'datasets/training/building.npy'
    with open(data_path, 'rb') as file:
        training_data_np = np.load(file)
    print("Loaded data")
    training_data_torch = torch.from_numpy(training_data_np)

    data_loader = DataLoader(training_data_torch, batch_size=1024, shuffle = True)
    model = LiDAR_NeRF(hidden_dim=512, embedding_dim_dir=15).to(device)
    optimizer = torch.optim.Adam(model.parameters(),lr=5e-4)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[2, 4, 8, 16], gamma=0.5)
    losses = train(model, optimizer, scheduler, data_loader, num_epoch = 16, device=device)
    losses_np = np.array(losses)
    np.save('v4trial3_losses', losses_np)
    print("\nTraining completed")

    ### Save the model
    torch.save(model.state_dict(), 'local/models/version4_trial3.pth')