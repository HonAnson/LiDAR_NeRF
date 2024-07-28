import numpy as np
from visualize import getMask, quat2RotationMatrix
from utility import quickVizNumpy, quickVizTwoNumpy




if __name__ == "__main__":
    data_path = r'datasets/training_euclidean/building.npy'
    with open(data_path, 'rb') as file:
        training_data_np = np.load(file)
    # get poitns for creating mask
    points = (np.random.shuffle(training_data_np))[0:1000000, 6:9]

    camera_pos = [0,0,0]
    camera_dir = [1,0,0]

    FOCAL_LENGTH = 1.43580442
    IMG_HEIGHT = 800
    IMG_WIDTH = 800











