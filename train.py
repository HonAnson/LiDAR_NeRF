import numpy as np
# import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
# import torch.optim as optim
# import json
# from tqdm import tqdm       # for showing progress when training
# import open3d as o3d        # for getting point cloud register
from einops import rearrange, repeat
# from numpy import sin, cos
from utility import printProgress

# def getDirections(angles):
#     """ Convert torch tensor of angles to 
#     cartiseian coordinate unit vector pointing that direction
#     """
#     elev, pan = angles[:,0], angles[:,1]
#     x_tilde, y_tilde, z_tilde = cos(elev)*cos(pan), cos(elev)*sin(pan), sin(elev)      
#     unit_vectors = torch.vstack([x_tilde, y_tilde, z_tilde])
#     return unit_vectors

def getSpacing(num_points, num_bins):
    """return a [num_points*num_bins, 1] pytorch tensor
    """
    # create a list of magnitudes with even spacing from 0 to 1
    t = torch.linspace(0,1, num_bins).expand(num_points, num_bins)  # [batch_size, num_bins//2]
    
    # preterb the spacing
    mid = (t[:, :-1] + t[:, 1:]) / 2.
    lower = torch.cat((t[:, :1], mid), -1)
    upper = torch.cat((mid, t[:, -1:]), -1)
    u = torch.rand(t.shape)
    t = lower + (upper - lower) * u  # [batch_size, nb_bins//2]
    # hard code start and end value of spacing
    t = rearrange(t, 'a b -> (a b) 1')  # [num_bins*batch_size, 1] 
    return t  

def getSamplesAndTarget(centres, directions, points, num_bins = 100):
    num_points = centres.shape[0]
    centres_tiled = repeat(centres, 'n c-> (n b) c', b = num_bins)
    directions_tiled = repeat(directions, 'n c-> (n b) c', b = num_bins)
    relative_pos = points - centres
    depth = torch.sqrt((relative_pos**2).sum(1))
    depth_tiled = repeat(depth, 'n -> (n b) 1', b = num_bins)
    t = getSpacing(num_points, num_bins)
    sample_magnitudes = t*depth_tiled
    sample_pos = directions_tiled*sample_magnitudes + centres_tiled   # [num_bins*num_points, 3]

    # tile points too
    points_target = repeat(points, 'n c-> (n b) c', b = num_bins)
    return sample_pos, directions_tiled, points_target


def warpu2d(u, focus = torch.tensor(1)):
    """ Map value u between 0 to 1, to a depth between 0 to +inf"""
    # TODO: deal with u being too close to 0 or too close to 1
    # TODO: add "slope" for this function

    sigmoid = nn.Sigmoid()
    offset = sigmoid(-focus)
    d_netural = torch.logit(u*(1-offset) + offset)
    return d_netural + focus


class LiDAR_NeRF(nn.Module):
    def __init__(self, embedding_dim_pos = 5, embedding_dim_dir = 5, hidden_dim = 256, device = 'cuda'):
        super(LiDAR_NeRF, self).__init__()
        self.device = device
        self.embedding_dim_dir = embedding_dim_dir
        self.embedding_dim_pos = embedding_dim_pos
        self.block1 = nn.Sequential(
            nn.Linear(embedding_dim_pos * 6 + 3 + embedding_dim_dir * 6 + 3, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),               
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),               
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),               
        )
        
        self.block2 = nn.Sequential(
            nn.Linear(embedding_dim_pos * 6 + 3 + embedding_dim_dir * 6 + 3 + hidden_dim, hidden_dim), nn.ReLU(),               
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),               
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),               
            nn.Linear(hidden_dim, hidden_dim//2), nn.ReLU(),
            nn.Linear(hidden_dim//2, 1), nn.ReLU()
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


def train(model, optimizer, scheduler, dataloader, device = 'cuda', num_epoch = int(1e5), num_bins = 10):
    training_losses = []
    loss_MSE = nn.MSELoss()
    num_batch_in_data = len(dataloader)
    count = 0
    for epoch in range(num_epoch):
        for iter, batch in enumerate(dataloader):

            # parse the batch
            laser_org = batch[:,0:3]
            laser_dir = batch[:,3:6]
            points = batch[:,6:9]

            sample_pos, sample_dir, points_target = getSamplesAndTarget(laser_org, laser_dir, points, num_bins=num_bins)
            sample_pos = sample_pos.to(device, dtype = torch.float32)
            sample_dir = sample_dir.to(device, dtype = torch.float32)
            points_target = points_target.to(device, dtype = torch.float32)
            
            # inference
            laser_dist_pred = model(sample_pos, sample_dir)
            points_pred = sample_dir* laser_dist_pred + sample_pos

            loss = loss_MSE(points_pred,points_target)         # + lossEikonal
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            ### Prin loss messages
            if count % 500 == 0:
                training_losses.append(loss.item())
            count += 1
            message = f"Training model... epoch: ({epoch}/{num_epoch}) | iteration: ({iter}/{num_batch_in_data}) | loss: {loss.item()}"
            printProgress(message)

        scheduler.step()
    return training_losses



if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using {device} device")

    data_path = r'datasets/training_euclidean/building.npy'
    with open(data_path, 'rb') as file:
        training_data_np = np.load(file)
    print("Loaded data")
    
    training_data_torch = torch.from_numpy(training_data_np)
    data_loader = DataLoader(training_data_torch, batch_size=2048, shuffle = True)
    model = LiDAR_NeRF(hidden_dim=512, embedding_dim_dir=5).to(device)
    optimizer = torch.optim.Adam(model.parameters(),lr=5e-4)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[2, 4, 8, 16], gamma=0.5)

    # Train model
    ver_name = "ver_euclidean_trial0"
    print("\nTraining version: " + ver_name)
    losses = train(model, optimizer, scheduler, data_loader, num_epoch = 8, device=device)
    losses_np = np.array(losses)
    np.save(ver_name + '_losses', losses_np)
    print("\nTraining completed")

    ### Save the model
    torch.save(model.state_dict(), 'local/models/' + ver_name + '.pth')