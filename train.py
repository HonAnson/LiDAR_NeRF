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

    # apply inverse sigmoid function to even spacing
    t = torch.log(t / (1 - t) + 10e-8)  
    t = rearrange(t, 'a b -> (a b) 1')  # [num_bins, batch_size]  transpose for multiplication broadcast
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
    centres_tiled = torch.tensor(repeat(centres, 'n c -> (n b) c', b = num_bins)) # [num_bin*batch_size, 3]
    # complete getting sample position by adding camera centre position to sampled position
    pos = centres_tiled + pos

    # tile the angle too
    angles_tiled = torch.tensor(repeat(angles, 'n c -> (n b) c', b = num_bins))
    return pos, angles_tiled, centres_tiled



def getUpSamples(origins, angles, gt_distance, num_rolls = 1):
    upsample_pos = torch.empty(0,3)
    upsample_ang = torch.empty(0,2)
    upsample_gt_dist = torch.empty(0,1)

    for num_roll in range(1, num_rolls+1):
        # first, we prepare pairs of data, where one of them has a shorter ray, and another has a longer ray
        gt_distance_rolled = torch.roll(gt_distance, num_roll, 0)
        condition =  gt_distance < gt_distance_rolled
        condition = rearrange(condition, 'a -> a 1')

        dir = torch.tensor(sph2cart(angles))
        gt_dist = rearrange(gt_distance, 'a -> a 1')
        gt_distance_rolled = rearrange(gt_distance_rolled, 'a -> a 1')
        pos = gt_dist * dir

        pos_shorter = torch.where(condition, pos, torch.roll(pos, num_roll, 0))
        origins_shorter = torch.where(condition, origins, torch.roll(origins, num_roll, 0))
        angles_shorter = torch.where(condition, angles, torch.roll(angles, num_roll, 0))
        gt_dist_shorter = torch.where(condition, gt_dist, gt_distance_rolled)

        # pos_longer = torch.where(condition, torch.roll(pos, num_roll, 0), pos)
        origins_longer = torch.where(condition, torch.roll(origins, num_roll, 0), origins)
        angles_longer = torch.where(condition, torch.roll(angles, num_roll, 0), angles)
        # gt_dist_longer = torch.where(condition, gt_distance_rolled, gt_dist)
        
        # check if angle between pairs are small
        diff = torch.abs(angles_shorter - angles_longer)
        mask_small_ang = (diff < 0.09).all(dim=1)   ### NOTE: hardcoded 0.09 radient difference max

        # check if origin between pairs are small
        diff2 = torch.abs(origins_shorter - origins_longer)
        mask_small_org = (diff2 < 0.2).all(dim=1)   ### pass if coordinate in origin are less than 20cm in all dimensions
        
        # get masks for both cases
        mask_same_org = mask_small_ang & mask_small_org

        ### Handling case of same origin
        # prepare set where rays are to be upsampled 
        # ensuring samples are at rays that are longer
        angles_from_same_org = angles_longer[mask_same_org]
        origins_from_same_org = origins_longer[mask_same_org]
        gt_dist_to_same_org = gt_dist_shorter[mask_same_org] 
        pos_to_same_org = pos_shorter[mask_same_org]

        if angles_from_same_org.shape[0] == 0:
            continue   # skip if there are no points available for upsampling

        # calculate upsampling position
        num_bins = 20 
        t = torch.linspace(0,1, num_bins).expand(angles_from_same_org.shape[0], num_bins)  # [batch_size, num_bins]
        
        # preterb the spacing
        mid = (t[:, :-1] + t[:, 1:]) / 2.
        lower = torch.cat((t[:, :1], mid), -1)
        upper = torch.cat((mid, t[:, -1:]), -1)
        u = torch.rand(t.shape)
        t = lower + (upper - lower) * u  # [batch_size, nb_bins]
        t = rearrange(t, 'a b -> b a')  # [num_bins, batch_size]  take transpose so that multiplication can broadcast
        t = rearrange(t, 'a b -> (b a) 1')
        t = torch.sqrt(t)
        
        # get the sampling positions
        origins_from_tiled = repeat(origins_from_same_org, 'n c -> (n b) c', b = num_bins)
        dir_from = torch.tensor(sph2cart(angles_from_same_org))
        dir_from_tiled = repeat(dir_from, 'n c -> (n b) c', b = num_bins)
        gt_dist_to_tiled = repeat(gt_dist_to_same_org, 'n c -> (n b) c', b = num_bins)     

        pos_from = origins_from_tiled + dir_from_tiled * t * gt_dist_to_tiled
        pos_to_tiled = repeat(pos_to_same_org, 'n c -> (n b) c', b = num_bins)

        # also calculte the ground truth distance of our up sampled location
        sample_sph = cart2sph(pos_from - pos_to_tiled)
        sample_direction = torch.tensor(sample_sph[:,1:])
        sample_gt_distance = torch.tensor(sample_sph[:,0])

        # add one more dimension to sample_gt_distance
        sample_gt_distance = rearrange(sample_gt_distance, 'a -> a 1')

        upsample_pos = torch.vstack((upsample_pos, pos_from))
        upsample_ang = torch.vstack((upsample_ang, sample_direction))
        upsample_gt_dist = torch.vstack((upsample_gt_dist, sample_gt_distance))

    # return pos_from , torch.tensor(sample_direction), torch.tensor(sample_gt_distance)
    return upsample_pos, upsample_ang, upsample_gt_dist


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
            nn.Linear(embedding_dim_pos * 6 + 3 + embedding_dim_dir * 4 + 2, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.Sigmoid(),               
            nn.Linear(hidden_dim, hidden_dim), nn.Sigmoid(),               
            nn.Linear(hidden_dim, hidden_dim), nn.Sigmoid(),               
        )
        
        self.block2 = nn.Sequential(
            nn.Linear(embedding_dim_pos * 6 + 3 + embedding_dim_dir * 4 + 2 + hidden_dim, hidden_dim), nn.ReLU(),               
            nn.Linear(hidden_dim, hidden_dim), nn.Sigmoid(),               
            nn.Linear(hidden_dim, hidden_dim), nn.Sigmoid(),               
            nn.Linear(hidden_dim, hidden_dim), nn.Sigmoid(),
            nn.Linear(hidden_dim,1)
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
        input = torch.hstack((emb_x,emb_d)).to(dtype=torch.float32)
        temp = self.block1(input)
        input2 = torch.hstack((temp, input)).to(dtype=torch.float32) # add skip input
        output = self.block2(input2)
        return output


def train(model, optimizer, scheduler, dataloader, device = 'cuda', num_epoch = int(1e5), num_bins = 100):
    training_losses = []
    num_batch_in_data = len(dataloader)
    for epoch in range(num_epoch):
        for iter, batch in enumerate(dataloader):

            # parse the batch
            ground_truth_distance = batch[:,0]
            angles = batch[:,1:3]
            origins = batch[:,3:6]

            sample_pos, sample_ang, sample_org = getSamples(origins, angles, ground_truth_distance, num_bins=num_bins)
            upsample_pos, upsample_ang, upsample_gt_distance = getUpSamples(origins, angles, ground_truth_distance, num_rolls=0)
            # tile distances
            gt_distance_tiled = repeat(ground_truth_distance, 'b -> (b n) 1', n=num_bins)

            # stack the upsampled position to sampled positions
            pos = (torch.vstack((sample_pos, upsample_pos))).to(device)
            ang = (torch.vstack((sample_ang, upsample_ang))).to(device)
            gt_dis = (torch.vstack((gt_distance_tiled,upsample_gt_distance))).to(device)
            org = (torch.vstack((sample_org, upsample_pos))).to(device)
            
            # inference
            rendered_value = model(pos, ang)
            sigmoid = nn.Sigmoid()
            rendered_value_sigmoid = sigmoid(rendered_value)
            actual_value_sigmoided = (getTargetValues(pos, gt_dis, org)).to(dtype = torch.float32)
            # loss = lossBCE(rendered_value, actual_value_sigmoided)  # + lossEikonal(model)

            # back propergate
            # print("iter: ", iter)
            # print("max rendered: ", max(rendered_value_sigmoid))
            # print("min rendered: ", min(rendered_value_sigmoid))
            # print("max gt: ", max(actual_value_sigmoided))
            # print("min gt: ", min(actual_value_sigmoided))
            
            loss_bce = nn.BCELoss()
            loss = loss_bce(rendered_value_sigmoid, actual_value_sigmoided)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            training_losses.append(loss.item())
            message = f"training model... epoch: ({epoch}/{num_epoch}) | iteration: ({iter}/{num_batch_in_data}) | loss: {loss.item()})"
            printProgress(message)

        scheduler.step()
    return training_losses




def runTrain():
    pass
    return


if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using {device} device")

    data_path = r'datasets/training/box_plant2.npy'
    with open(data_path, 'rb') as file:
        training_data_np = np.load(file)
    print("loaded data")
    training_data_torch = torch.from_numpy(training_data_np)

    data_loader = DataLoader(training_data_torch, batch_size=1024, shuffle = True)
    model = LiDAR_NeRF(hidden_dim=512, embedding_dim_dir=15).to(device)
    optimizer = torch.optim.Adam(model.parameters(),lr=5e-4)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[2, 4, 8, 16], gamma=0.5)
    losses = train(model, optimizer, scheduler, data_loader, num_epoch = 8, device=device)

    ### Save the model
    # torch.save(model.state_dict(), 'local/models/version4_trial1.pth')