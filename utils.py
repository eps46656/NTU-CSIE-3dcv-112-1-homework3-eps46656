import cv2 as cv
import numpy as np
import os
import sys

DIR = os.path.dirname(__file__).replace("\\", "/")

sys.path.append(DIR)

RAD_TO_DEG = 180 / np.pi
DEG_TO_RAD = np.pi / 180

# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------

bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)

def GetMatches2(des1, des2, k=1024):
    matches = bf.match(des1, des2)

    matches = sorted(matches, key=lambda m: m.distance)
    matches = matches[:min(len(matches), k)]

    return np.array([
        [matches.queryIdx, matches.trainIdx] for matches in matches],
        dtype=np.uint)

def MergeMatches(matches12, matches23):
    matches23 = {m[0]: m[1] for m in matches23}

    ret = list()

    for m in matches12:
        idx1 = m[0]
        idx2 = m[1]
        idx3 = matches23.get(idx2, None)

        if idx3 is not None:
            ret.append((idx1, idx2, idx3))

    return np.array(ret, dtype=np.uint)

def GetCameraCone(H, W, camera_mat, RT):
    camera_mat_inv = np.linalg.inv(camera_mat)

    og = np.array([[0], [0], [0], [1]])
    tl = np.zeros((4, 1))
    tr = np.zeros((4, 1))
    br = np.zeros((4, 1))
    bl = np.zeros((4, 1))

    tl[:3, :] = camera_mat_inv @ np.array([[  0], [  0], [1]])
    tr[:3, :] = camera_mat_inv @ np.array([[W-1], [  0], [1]])
    br[:3, :] = camera_mat_inv @ np.array([[W-1], [H-1], [1]])
    bl[:3, :] = camera_mat_inv @ np.array([[  0], [H-1], [1]])

    og = (RT @ og)[:3, 0]
    tl = og + (RT @ tl)[:3, 0]
    tr = og + (RT @ tr)[:3, 0]
    br = og + (RT @ br)[:3, 0]
    bl = og + (RT @ bl)[:3, 0]

    return og, tl, tr, br, bl

def GetAng(u, v):
    return np.arccos(
        np.dot(u, v) / np.linalg.norm(u) / np.linalg.norm(v))

def GetRatio(pc1, pc2):
    # pc1 = {"t1": t1,
    #        "t2": t2,
    #        "x1": x1,
    #        "x2": x2,
    #        "x": x,
    #        "err": err,}
    # pc2 = {"t1": t1,
    #        "t2": t2,
    #        "x1": x1,
    #        "x2": x2,
    #        "x": x,
    #        "err": err,}

    N, P = pc1["x"].shape

    ok_pc = list()

    ERR = 0.05

    for i in range(N):
        pc1_t1 = pc1["t1"][i]
        pc1_t2 = pc1["t2"][i]
        pc2_t1 = pc2["t1"][i]
        pc2_t2 = pc2["t2"][i]

        if pc1_t1 < 0 or pc1_t2 < 0 or pc2_t1 < 0 or pc2_t2 < 0:
            pass

        pc1_err = pc1["err"][i]
        pc2_err = pc2["err"][i]

        if ERR <= pc1_err or ERR <= pc2_err:
            pass

        pc1_x = pc1["x"][i]
        pc2_x = pc2["x"][i]

        ang_err = GetAng(pc1_x, pc2_x)

        if 4 * DEG_TO_RAD <= ang_err:
            pass

        ok_pc.append((ang_err, pc1_x, pc2_x))

    ok_pc = sorted(ok_pc, key=lambda t: t[0])

    return np.linalg.norm(ok_pc[0][2] - ok_pc[1][1]) / \
           np.linalg.norm(ok_pc[0][2] - ok_pc[1][1])

def EstimateRT(points1, points2, camera_mat):
    # points1[N, 2]
    # points2[N, 2]

    N = points1.shape[0]

    assert points1.shape == (N, 2)
    assert points2.shape == (N, 2)

    E, mask = cv.findEssentialMat(
        points1=points1,
        points2=points2,
        cameraMatrix=camera_mat)

    num_of_points, R_, t_, mask = cv.recoverPose(
        E=E,
        points1=points1,
        points2=points2,
        cameraMatrix=camera_mat)

    mask = mask.reshape((-1,)) != 0

    RT = np.eye(4)
    RT[:3, :3] = R_
    RT[:3, 3] = t_[:, 0]

    return np.linalg.inv(RT), mask

def GetIntersections(o1, d1, o2, d2):
    # o1[N, 3]
    # d1[N, 3]
    # o2[N, 3]
    # d2[N, 3]

    # consider two lines
    # x1 = o1 + t1 * d1
    # x2 = o2 + t2 * d2

    # find x which minimize
    # (o1 + t1 * d1 - x)^2
    # (o2 + t2 * d2 - x)^2

    N = o1.shape[0]

    assert o1.shape == (N, 3)
    assert d1.shape == (N, 3)
    assert o2.shape == (N, 3)
    assert d2.shape == (N, 3)

    d1d1 = (d1 * d1).sum(-1) # [N]
    d1d2 = (d1 * d2).sum(-1) # [N]
    d2d2 = (d2 * d2).sum(-1) # [N]

    o2o1 = o2 - o1 # [N, 3]

    M = np.empty([N, 2, 2])
    M[:, 0, 0] =  d1d1
    M[:, 0, 1] = -d1d2
    M[:, 1, 0] =  d1d2
    M[:, 1, 1] = -d2d2

    k = np.empty([N, 2, 1])
    k[:, 0, 0] = (d1 * o2o1).sum(-1)
    k[:, 1, 0] = (d2 * o2o1).sum(-1)

    t = np.linalg.inv(M) @ k # [N, 2, 1]

    t1 = t[:, 0, :] # [N, 1]
    t2 = t[:, 1, :] # [N, 1]

    x1 = o1 + d1 * t1 # [N, 3]
    x2 = o2 + d2 * t2 # [N, 3]

    dx = x2 - x1 # [N, 3]

    err = (dx**2).sum(axis=1)**0.5 # [N]

    x = (x1 + x2) / 2 # [N, 3]

    assert t1.shape == (N, 1)
    assert t2.shape == (N, 1)
    assert x.shape == (N, 3)

    return t1, t2, x1, x2, x, err

def ConstructPoints(camera_mat, points1, points2, dRT):
    # points1[N, 2]
    # points2[N, 2]
    # dRT[4, 4]

    N = points1.shape[0]

    assert points1.shape == (N, 2)
    assert points2.shape == (N, 2)
    assert dRT.shape == (4, 4)

    camera_mat_inv = np.linalg.inv(camera_mat)

    '''
    p1 = lambda * K * x1
       = lambda * K * x1

    p2 = lambda * K * dR^-1 * (x2 - dt)
       = lambda * K * dR^-1 * x2 - lambda * K * dR^-1 * dt

    x1 = t1 * (K^-1 * p1)

    x2 = dR * K^-1 * (t2 * p2 + K * dR^-1 * dt)
       = t2 * (dR * K^-1 * p2) + dt
    '''

    p1 = np.empty((3, N))
    p1[:2, :] = points1.transpose()
    p1[2, :] = 1

    p2 = np.empty((3, N))
    p2[:2, :] = points2.transpose()
    p2[2, :] = 1

    o1 = np.zeros((N, 3)) # [N, 3]
    d1 = (camera_mat_inv @ p1).transpose() # [N, 3]

    dR = dRT[:3, :3]
    dt = dRT[:3, 3]

    o2 = dt.reshape([1, 3]).repeat(N, axis=0) # [N, 3]
    d2 = (dR @ camera_mat_inv @ p2).transpose() # [N, 3]

    t1, t2, x1, x2, x, err = GetIntersections(o1, d1, o2, d2)

    return {"t1": t1,
            "t2": t2,
            "x1": x1,
            "x2": x2,
            "x": x,
            "err": err,}
