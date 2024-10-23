import cv2 as cv
import numpy as np
import os
import sys

DIR = os.path.dirname(__file__).replace("\\", "/")

sys.path.append(DIR)
from utils import *

def main():
    N = 23

    o1 = np.random.rand(N, 3)
    d1 = np.random.rand(N, 3)
    o2 = np.random.rand(N, 3)
    d2 = np.random.rand(N, 3)

    t1, t2, x = GetIntersections(o1, d1, o2, d2)

    avg_dist = GetAvgDist(x)

    print(f"avg_dist = {avg_dist}")


if __name__ == "__main__":
    main()
