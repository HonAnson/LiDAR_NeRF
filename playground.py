import numpy as np
from visualize import getMask, quat2RotationMatrix
from utility import quickVizNumpy, quickVizTwoNumpy
import cv2



if __name__ == "__main__":
    data_path = r'datasets/training_euclidean/round_plant2.npy'
    with open(data_path, 'rb') as file:
        training_data_np = np.load(file)
    # get poitns for creating mask
    np.random.shuffle(training_data_np)
    points = training_data_np[0:1000000, 6:9] / 0.024429254132039887
    camera_pos = np.array([0,0,0])
    camera_dir = np.array([1,0,0])

    FOCAL_LENGTH = 1.43580442   # unit = meter
    IMG_HEIGHT = 800
    IMG_WIDTH = 800

    mask = getMask(points, camera_pos, camera_dir, FOCAL_LENGTH, IMG_HEIGHT, IMG_WIDTH)
    matrix_normalized = (mask * 255).astype(np.uint8)
    cv2.imshow('Mask', matrix_normalized)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


    # quickVizNumpy(to_viz)














