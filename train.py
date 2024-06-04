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


# load data from dataset directory, filterout irralevent datas
def loadData():
    # Specify the directory path
    dataset_path = 'datasets/testing1'

    # List all files in the specified path, ignoring directories
    files = [f for f in os.listdir(dataset_path) if os.path.isfile(os.path.join(dataset_path, f))]
    files.sort()

    # read the files
    points_xyz = []
    for s in files:
        path = 'datasets/testing1/' + s
        df = pd.read_csv(path)
        a = df.to_numpy()
        b = a[:,8:11]
        points_xyz.append(b)
    return points_xyz   # type = list


### we now process the data
# (1) Convert data to spherical coordinate
# (2) Filter out extreme value datas
# 3. Stack them into one big numpy matrix
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

    # camera centre locations
    centres = [-t for t in translations]
    centres_data = []
    for i,c in enumerate(centres):
        l = len(points_sphere[i])
        temp = np.tile(c, (l, 1))
        centres_data.append(temp)

    # (3) Stack camera centre, r and angles into a matrix
    stacked = []
    for i in range(len(points_sphere)):
        temp = np.hstack((points_sphere[i], centres_data[i]))
        stacked.append(temp)

    dataset = np.array([])
    for i in range(len(stacked)):
        if i == 0:
            dataset = stacked[i]
        else:
            dataset = np.vstack((dataset, stacked[i]))
    np.random.shuffle(dataset)

    # (2) Filter out points where the distance value is = 0
    mask1 = dataset[:, 0] != 0
    mask2 = dataset[:,0] > 50
    mask = mask1 + mask2
    dataset = dataset[mask]
    return dataset   # type = numpyArray



### Train the model
def trainModel(dataset):

    features = 5
    batch_size = 256

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using {device} device")

    # Convert dataset to pytorch tensor
    X = np.array(dataset[:,1:])
    y = np.array(dataset[:,0])

    # Convert to tensor:
    X_tensor = torch.from_numpy(X).double()
    y_tensor = torch.from_numpy(y).double()

    dataset = TensorDataset(X_tensor, y_tensor)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Now we prepare to train the model

    class MLP(nn.Module):
        def __init__(self):
            super(MLP, self).__init__()
            self.layers = nn.Sequential(
                nn.Linear(5, 512),  # Input layer with 5 inputs and 10 outputs
                nn.ReLU(),                # Activation function
                nn.Linear(512, 512),        # Hidden layer with 512 neurons
                nn.ReLU(),                # Activation function
                nn.Linear(512, 512),        # Hidden layer with 512 neurons
                nn.ReLU(),                # Activation function
                nn.Linear(512, 512),        # Hidden layer with 512 neurons
                nn.ReLU(),                # Activation function
                nn.Linear(512, 512),        # Hidden layer with 512 neurons
                nn.ReLU(),                # Activation function
                nn.Linear(512, 1)          # Output layer with 1 output
            )
            
        def forward(self, x):
            return self.layers(x)

    # Initialize the model
    model = MLP().to(device)

    # Loss and Optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    # Training Loop
    num_epochs = 50
    for epoch in range(num_epochs):
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device).float(), targets.to(device).float()
            
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(loss)


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
    return rearrange(output, 'a b -> b a') 
#######################################




if __name__ == "__main__":
    points = loadData()
    dataset = prepareData(points)
    trainModel(dataset)