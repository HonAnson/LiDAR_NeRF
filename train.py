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
    """return a [num_points, num_bins] pytorch tensor
    Give perturbated even spacing value from 0 to 1 along num_bin dimension
    """
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
    return t


def invSigmoid(t, dist, sampling_variance):
    t = torch.log(t / ((1 - t) + 1e-8))
    dist_tiled = repeat(dist, 'p -> p b', b = t.shape[1])
    magnitude = t * sampling_variance + dist_tiled
    return magnitude

def getSamplingPositions(centres, directions, distance, sampling_variance, num_bins = 100, num_points = 1024):
    # apply inverse sigmoid function to even spacing
    t = getSpacing(num_points, num_bins)  # [num_points, num_bins]
    magnitudes = invSigmoid(t, distance, sampling_variance)
    delta = torch.cat((magnitudes[:, 1:] - magnitudes[:, :-1], torch.tensor([1e10]).expand(num_points, 1)), -1) # work on getting delta

    # reshape for calculation
    magnitudes_tiled = rearrange(magnitudes, 'n b -> (n b) 1')
    dir_tiled = repeat(directions, 'n c -> (n b) c', b = num_bins)
    centres_tiled = repeat(centres, 'n c -> (n b) c', b = num_bins)
    pos = magnitudes_tiled*dir_tiled + centres_tiled      # p = d*t + o

    # reshape sampled position back to [num_points, num_bins, 3]
    pos = rearrange(pos, '(n b) c -> n b c', n = num_points, b = num_bins)
    return pos, delta, magnitudes

def computeCumulativeTransmittance(alpha, device):
    # alpha = 1 - exp(-density * delta)
    K = torch.cumprod((1 - alpha), 1)   # K is just a temporary variable
    ones = torch.ones((K.shape[0],1), device = device)
    T = torch.cat((ones, K[:,:-1]), 1)
    return T

def computeTerminationDistribution(T, alpha):
    h_temp = T * alpha
    return h_temp

def computeExpectedDepth(h, magnitude):
    d_hat = torch.sum(h * magnitude, axis = 1)
    return d_hat

def getTargetCumulativeTransmittance(magnitude, distance, prediction_variance, device):
    # add one dimension to distance for calculation
    distance_temp = (rearrange(distance, 'p -> p 1')).to(device = device, dtype = torch.float32)
    t = magnitude - distance_temp
    sigmoid = nn.Sigmoid()
    target_T = sigmoid(-t/prediction_variance)    # note that 1 - sigmoid(x) = sigmoid(-x). And I am modelling cumulative transmittance as 1-sigmoid(x), centred at lidar measurement
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
            nn.Linear(hidden_dim//2,1), nn.Softplus()      # relu for last layer, as we are predicting densities
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


def train(model, optimizer, scheduler, dataloader, device = 'cuda', num_epoch = int(1e5), num_bins = 100, sampling_variance = 0.5, prediction_variance = 0.1, lambda_T = 1, lambda_d = 1, lambda_h = 1):
    training_losses = []
    num_batch_in_data = len(dataloader)
    count = 0
    KL_loss = nn.KLDivLoss(reduction = 'batchmean')
    MSE_loss = nn.MSELoss()
    BCE_loss = nn.BCELoss()
    breakpoint()
    for epoch in range(num_epoch):
        for iter, batch in enumerate(dataloader):

            # parse the batch
            num_points = batch.shape[0]
            laser_org = batch[:,0:3]
            laser_dir = batch[:,3:6]
            distance = batch[:,6]

            # prepare sampling positions
            sample_pos, delta, magnitude = getSamplingPositions(laser_org, laser_dir, distance, sampling_variance, num_bins=num_bins, num_points = num_points)
            
            # transfer tensors to device
            delta = delta.to(device, dtype = torch.float32)  # [num_points, num_bin]
            magnitude = magnitude.to(device, dtype = torch.float32)  # [num_points, num_bin]
            sample_pos = sample_pos.to(device, dtype = torch.float32)  # [num_points, num_bin, 3]

            # inference
            input_pos = rearrange(sample_pos, 'n b c -> (n b) c')
            density_pred = model(input_pos)
            density_pred = rearrange(density_pred, '(n b) 1 -> n b', n = num_points, b = num_bins)

            # compute cumulative transmittance from density prediction
            alpha = 1 - torch.exp(-(density_pred + 1e-6) * delta)
            T_pred = computeCumulativeTransmittance(alpha, device)
            h_pred = computeTerminationDistribution(T_pred, alpha)
            h_pred += 1e-8  # avoid under flow
            h_pred /= rearrange(h_pred.sum(1), 'n -> n 1')  # make sure prediction is a valid distribution
            d_pred = computeExpectedDepth(h_pred, magnitude)

            # also get target values
            T_target = getTargetCumulativeTransmittance(magnitude, distance, prediction_variance, device = device)
            h_target = getTargetTerminationDistribution(T_target, delta, prediction_variance)
            d_target = distance.to(device, dtype = torch.float32)

            # calculate losses 
            loss_T = BCE_loss(T_pred,T_target)
            loss_h = KL_loss(h_pred.log(), h_target)        #taking log because pytorch assume predicted value to be in log space
            loss_d = MSE_loss(d_pred, d_target)     
            loss_together = lambda_T * loss_T + lambda_d * loss_d + lambda_h * loss_h     # TODO: hyperparameter tunning

            optimizer.zero_grad()
            loss_together.backward()
            optimizer.step()
            ### Print loss messages and store losses
            if count % 500 == 0:
                training_losses.append(loss_together.item())
            count += 1
            message = f"Training model... epoch: ({epoch}/{num_epoch}) | iteration: ({iter}/{num_batch_in_data}) | loss_T: {loss_T.item():.4f} | loss_h: {loss_h.item():.4f} | loss_d: {loss_d.item():.4f} | loss: {loss_together.item():.4f}"
            printProgress(message)
        scheduler.step()
    return training_losses




if __name__ == "__main__":
    ######################
    ### Job Parameters ###
    ######################
    JOB_NAME = r'ver_cumulative_trial0'
    DATA_NAME = r'round_plant2'

    ######################
    ## Hyper Parameters ##
    ######################
    HIDDEN_DIM = 512
    EMBEDDING_DIM_DIR = 10
    EMBEDDING_DIM_POS = 15
    LEARNING_RATE = 5e-6
    BATCH_SIZE = 1024
    NUM_EPOCH = 32
    LAMBDA_T = 50
    LAMBDA_d = 1
    LAMBDA_h = 1
    SAMPLING_VARIANCE = 0.5
    PREDICTION_VARIANCE = 0.1
    NUM_BINS = 100

    #########################
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using {device} device")
    
    data_path = r'datasets/training_cumulative/' + DATA_NAME + r'.npy'
    with open(data_path, 'rb') as file:
        training_data_np = np.load(file)
    print("Loaded data")
    
    training_data_torch = torch.from_numpy(training_data_np)
    data_loader = DataLoader(training_data_torch, batch_size= BATCH_SIZE, shuffle = True)
    model = LiDAR_NeRF(hidden_dim=HIDDEN_DIM, embedding_dim_dir=EMBEDDING_DIM_DIR, embedding_dim_pos=EMBEDDING_DIM_POS).to(device)
    optimizer = torch.optim.Adam(model.parameters(),lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[2, 4, 8, 16], gamma=0.5)
    losses = train(model, 
                   optimizer, 
                   scheduler, 
                   data_loader, 
                   num_epoch = NUM_EPOCH, 
                   device=device, 
                   lambda_T = LAMBDA_T, 
                   lambda_d = LAMBDA_d, 
                   lambda_h = LAMBDA_h, 
                   sampling_variance = SAMPLING_VARIANCE, 
                   prediction_variance = PREDICTION_VARIANCE)
                   
    losses_np = np.array(losses)

    ### Save output
    np.save('ver_cumulative_trial0_losses', losses_np)
    print("\nTraining completed")

    ### Save the model
    # torch.save(model.state_dict(), 'local/models/ver_cumulative_trial0.pth')
