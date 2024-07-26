import torch
import numpy as np
import matplotlib.pyplot as plt
import json
import open3d as o3d
import pandas as pd
import os
import copy
from utility import listFiles, quickVizNumpy
from scipy.spatial.transform import Rotation as R
from numpy import cos, sin, sqrt, arctan2, array
import torch.nn as nn
from einops import rearrange
### fucking around
from preprocess import loadDataFromRegisteredSlam, getDistanceAndDirection, prepareTrainingData
from train import getSpacing, getTargetCumulativeTransmittance, getTargetTerminationDistribution


name = r'round_plant2'
input_path = r'datasets/registered/' + name + r'.json'
data = loadDataFromRegisteredSlam(input_path)
temp = prepareTrainingData(data)

quickVizNumpy(temp[200000:600000,:])
breakpoint()









# variance = 0.1
# t, delta = getSpacing(2, 1000)
# T = getTargetCumulativeTransmittance(t, variance=variance)
# h = getTargetTerminationDistribution(T, delta, variance=variance)

# T[0,-1] = 0
# temp  = sum((T[0,:] * (1 - T[0,:]))*delta[0,:])
# breakpoint()
# x = np.array(t[0,:])
# y = np.array()

# plt.plot(x, k*(1-k))
# plt.title("Plot of NumPy Array")
# plt.xlabel("x-axis")
# plt.ylabel("y-axis")
# plt.show()

