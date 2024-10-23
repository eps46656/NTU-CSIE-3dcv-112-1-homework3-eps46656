import open3d as o3d
import numpy as np
import cv2 as cv
import sys, os, argparse, glob
import multiprocessing as mp
import time
import sys

DIR = os.path.dirname(__file__).replace("\\", "/")

sys.path.append(DIR)

from utils import *

# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------

def main():
    RTs = np.load(f"{DIR}/RTs.npy", allow_pickle=True)[()]["RTs"]

    H, W = 360, 640

    camera_params = np.load(f"{DIR}/camera_parameters.npy",
                            allow_pickle=True)[()]
    camera_mat = camera_params['K']
    camera_distort = camera_params['dist']

    vis = o3d.visualization.Visualizer()
    vis.create_window()

    for frame_i in range(RTs.shape[0]):
        RT = RTs[frame_i]

        camera_cone = o3d.geometry.LineSet()

        camera_cone.points = o3d.utility.Vector3dVector(
            [*GetCameraCone(H, W, camera_mat, RT)])

        camera_cone.lines = o3d.utility.Vector2iVector([
            [0, 1], [0, 2], [0, 3], [0, 4],
            [1, 2], [2, 3], [3, 4], [4, 1],])

        camera_cone.colors = o3d.utility.Vector3dVector([
            [1, 0, 0] for i in range(8)])

        vis.add_geometry(camera_cone)
        vis.poll_events()

    vis.run()
    vis.destroy_window()

if __name__ == '__main__':
    main()
