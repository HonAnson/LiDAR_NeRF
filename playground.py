import numpy as np
from visualize import getMask, quat2RotationMatrix, getPsudoImgVec
from utility import quickVizNumpy, quickVizTwoNumpy
import cv2



if __name__ == "__main__":
    camear_direction = np.array([1,0,0])
    pixel_to_meter = 500
    img_width = 500
    img_height = 500
    focal_length = 1.43580442
    vec = getPsudoImgVec(camear_direction, pixel_to_meter, img_width, img_height, focal_length)
    


    




















