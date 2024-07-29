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


t = getSpacing(5,20)



breakpoint()






