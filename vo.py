import open3d as o3d
import numpy as np
import cv2 as cv
import sys, os, argparse, glob
import multiprocessing as mp
import time

DIR = os.path.dirname(__file__).replace("\\", "/")

sys.path.append(DIR)

from utils import *

# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------

orb = cv.ORB_create()

class SimpleVO:
    def __init__(self, args):
        camera_params = np.load(args.camera_parameters, allow_pickle=True)[()]
        self.camera_mat = camera_params['K']
        self.camera_distort = camera_params['dist']

        self.frame_paths = sorted(list(glob.glob(os.path.join(args.input, '*.png'))))

    def run(self):
        def Undistort(points):
            return cv.undistortPoints(
                points, self.camera_mat, self.camera_distort,
                None, self.camera_mat).reshape((-1, 2))

        H, W = 360, 640

        vis = o3d.visualization.Visualizer()
        vis.create_window()

        RT = np.eye(4, 4)

        kp1, des1 = None, None
        kp2, des2 = None, None
        kp3, des3 = None, None

        dRT12 = None
        dRT23 = None

        matches12 = None
        matches23 = None

        RTs = list()

        for frame_i in range(len(self.frame_paths)):
            frame_path = self.frame_paths[frame_i]
            img = cv.imread(frame_path)

            kp1, des1 = kp2, des2
            kp2, des2 = kp3, des3
            kp3, des3 = orb.detectAndCompute(img, None)
            kp3 = Undistort(np.array([kp.pt for kp in kp3]))

            if kp2 is None:
                dRT12 = dRT23
                dRT23 = np.array([
                    [1, 0, 0, 0],
                    [0, 1, 0, 0],
                    [0, 0, 1, 0],
                    [0, 0, 0, 1],])
            else:
                matches12 = matches23
                matches23 = GetMatches2(des2, des3)

                dRT12 = dRT23
                dRT23, mask23 = EstimateRT(
                    kp2[matches23[:, 0]],
                    kp3[matches23[:, 1]],
                    self.camera_mat)

                if kp1 is not None:
                    matches123 = MergeMatches(matches12, matches23)

                    points1 = kp1[matches123[:, 0]]
                    points2 = kp2[matches123[:, 1]]
                    points3 = kp3[matches123[:, 2]]

                    pc21 = ConstructPoints(
                        self.camera_mat, points2, points1, np.linalg.inv(dRT12))

                    pc23 = ConstructPoints(
                        self.camera_mat, points2, points3, dRT23)

                    dRT23[:3, 3] *= GetRatio(pc21, pc23)

            RT = RT @ dRT23

            RTs.append(RT)

            camera_cone = o3d.geometry.LineSet()

            camera_cone.points = o3d.utility.Vector3dVector(
                [*GetCameraCone(H, W, self.camera_mat, RT)])

            camera_cone.lines = o3d.utility.Vector2iVector([
                [0, 1], [0, 2], [0, 3], [0, 4],
                [1, 2], [2, 3], [3, 4], [4, 1],])

            camera_cone.colors = o3d.utility.Vector3dVector([
                [1, 0, 0] for i in range(8)])

            vis.add_geometry(camera_cone)
            vis.poll_events()

            cv.imshow("img", img)
            cv.waitKey(30)

            print(f"t =\n{RT[:3, 3]}")
            print(f"RT =\n{RT}")

        np.save("RTs.npy", {"RTs": np.array(RTs)})

        vis.run()
        vis.destroy_window()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('input', help='directory of sequential frames')
    parser.add_argument('--camera_parameters', default='camera_parameters.npy', help='npy file of camera parameters')
    args = parser.parse_args()

    vo = SimpleVO(args)
    vo.run()
