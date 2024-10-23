import cv2 as cv
import numpy as np
import os
import sys

DIR = os.path.dirname(__file__).replace("\\", "/")

sys.path.append(DIR)

from utils import *

# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------

def main():
    camera_params = np.load("camera_parameters.npy", allow_pickle=True)[()]
    camera_mat = camera_params['K']
    camera_distort = camera_params['dist']

    print(f"camera_mat")
    print(camera_mat)
    print(f"camera_distort")
    print(camera_distort)

    img1 = cv.imread(f"{DIR}/frames/frame_0010.png")
    img2 = cv.imread(f"{DIR}/frames/frame_0011.png")

    EstimateRT(img1, img2, camera_mat, camera_distort)

if __name__ == "__main__":
    main()
