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



def getSpacing(num_points, num_bins, variance = 0.1):
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
    t[:,0], t[:,-1] = 1e-3, 0.999
    # apply inverse sigmoid function to even spacing
    t = torch.log(t / ((1 - t) + 1e-8))

    t /= 13.8136 # constant obtained by invsigmoid(0.999)*2, which normalize t to between -0.5 to 0.5
    delta = torch.cat((t[:, 1:] - t[:, :-1], torch.tensor([10]).expand(num_points, 1)), -1)
    # t = rearrange(t, 'a b -> (a b) 1')  # [num_bins*batch_size, 1]
    # delta = rearrange(delta, 'a b -> (a b) 1')  # [num_bins*batch_size, 1]
    return t , delta

def getSamplingPositions(centres, directions, distance, t, num_bins = 100):
    dist_tiled = repeat(distance, 'n -> n b', b = num_bins)
    magnitudes = repeat(t + dist_tiled, 'n b -> n b c', c = 3)   # [num_points, num_bin, 3]
    dir_tiled = repeat(directions, 'n c -> n b c', b = num_bins)
    centres_tiled = repeat(centres, 'n c -> n b c', b = num_bins) # [num_points, num_bin, 3]
    pos = magnitudes*dir_tiled + centres_tiled
    return pos

def computeCumulativeTransmittance(alpha):
    T = torch.cumprod((1 - alpha), 1)
    return T

def computeTerminationDistribution(T, alpha):
    h = T * alpha
    return h

def computeExpectedDepth(h, sample_pos):
    d_hat = torch.sum(h * sample_pos, axis = 1)
    return d_hat

# returns pytorch tensor of sigmoid of projected SDF
def getTargetCumulativeTransmittance(t, variance = 1):
    sigmoid = nn.Sigmoid()
    target_T = sigmoid(-t/variance)    # note that 1 - sigmoid(x) = sigmoid(-x). And I am modelling cumulative transmittance as 1-sigmoid(x), centred at lidar measurement
    return target_T

def getTargetTerminationDistribution(target_T, delta, variance = 1):
    target_h = (target_T * (1 - target_T)) * (delta / variance)
    target_h[:,-1] = 0
    return target_h

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
            nn.Linear(hidden_dim, hidden_dim//2), nn.ReLU(),
            nn.Linear(hidden_dim//2,1), nn.ReLU()      # relu for last layer, as we are predicting densities
        )
        
    @staticmethod
    def positional_encoding(x, L):
        out = [x]
        for j in range(L):
            out.append(torch.sin(2 ** j * x))
            out.append(torch.cos(2 ** j * x))
        return torch.cat(out, dim=1)

    def forward(self, pos):
        emb_x = self.positional_encoding(pos, self.embedding_dim_pos)
        temp = self.block1(emb_x)
        temp2 = torch.cat((temp, emb_x), dim=1) # add skip input
        density = self.block2(temp2)
        return density


def train(model, optimizer, scheduler, dataloader, device = 'cuda', num_epoch = int(1e5), num_bins = 100, sampling_variance = 1, prediction_variance = 1):
    training_losses = []
    num_batch_in_data = len(dataloader)
    count = 0
    KL_loss = nn.KLDivLoss()
    MSE_loss = nn.MSELoss()
    #loss_bce = nn.BCELoss()
    for epoch in range(num_epoch):
        for iter, batch in enumerate(dataloader):

            # parse the batch
            num_points = batch.shape[0]
            centres = batch[:,0:3]
            directions = batch[:,3:6]
            distance = batch[:,6]

            # prepare sampling positions
            t, delta = getSpacing(num_points, num_bins)
            sample_pos = getSamplingPositions(centres, directions, distance, t, num_bins=num_bins)

            # transfer tensors to device
            t = t.to(device, dtype = torch.float32)  # [num_points, num_bin, 3]
            delta = delta.to(device, dtype = torch.float32)  # [num_points, num_bin, 3]
            sample_pos = sample_pos.to(device, dtype = torch.float32)  # [num_points, num_bin, 3]

            # inference
            input_pos = rearrange(sample_pos, 'n b c -> (n b) c')
            density_pred = model(input_pos)
            density_pred = rearrange(density_pred, '(n b) 1 -> n b', n = num_points, b = num_bins)

            # compute cumulative transmittance from density prediction
            alpha = 1 - torch.exp(-density_pred * delta)
            T_pred = computeCumulativeTransmittance(alpha)
            h_pred = computeTerminationDistribution(T_pred, delta)
            d_pred = computeExpectedDepth(h_pred, delta)

            # also get target values
            T_target = getTargetCumulativeTransmittance(t, variance = 0.1)
            h_target = getTargetTerminationDistribution(t, delta, variance=0.1)
            d_target = distance.to(device, dtype = torch.float32)

            # calculate losses 
            loss_T = MSE_loss(T_pred,T_target)
            # loss_h = KL_loss(h_pred, h_target)    # TODO: bug here, returning NAN, check DS NeRF source code for more
            loss_d = MSE_loss(d_pred, d_target)     # TODO: bug here too, model not converging
            loss_together = 10*loss_T + loss_d        # TODO: hyperparameter tunning
            optimizer.zero_grad()
            loss_together.backward()
            optimizer.step()
            ### Print loss messages and store losses
            if count == 2:
                breakpoint()
            if count % 500 == 0:
                training_losses.append(loss_together.item())
            count += 1
            message = f"Training model... epoch: ({epoch}/{num_epoch}) | iteration: ({iter}/{num_batch_in_data}) | loss: {loss_together.item()}"
            printProgress(message)

        scheduler.step()
    return training_losses




if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using {device} device")
    ####
    # Choose data here
    data_path = r'datasets/training_cumulative/building.npy'
    ####
    with open(data_path, 'rb') as file:
        training_data_np = np.load(file)
    print("Loaded data")
    training_data_torch = torch.from_numpy(training_data_np)

    data_loader = DataLoader(training_data_torch, batch_size=1024, shuffle = True)
    model = LiDAR_NeRF(hidden_dim=512, embedding_dim_dir=15).to(device)
    optimizer = torch.optim.Adam(model.parameters(),lr=5e-4)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[2, 4, 8, 16], gamma=0.5)
    losses = train(model, optimizer, scheduler, data_loader, num_epoch = 16, device=device, variance = 1)
    losses_np = np.array(losses)

    # np.save('ver_cumulative_trial0_losses', losses_np)
    # print("\nTraining completed")

    # ### Save the model
    # torch.save(model.state_dict(), 'local/models/ver_cumulative_trial0.pth')