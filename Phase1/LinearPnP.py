import numpy as np
from numpy import argmin

def LinearPnP(world_points, feature_points, K):
    A = []

    for i in range(len(world_points)):
        
        X = world_points[i][0]
        Y = world_points[i][1]
        Z = world_points[i][2]
        u = feature_points[i][0]
        v = feature_points[i][1]

        A.append([X, Y, Z, 1, 0, 0, 0, 0, -u*X, -u*Y, -u*Z, -u])
        A.append([0, 0, 0, 0, X, Y, Z, 1, -v*X, -v*Y, -v*Z, -v])

    u, s, vh = np.linalg.svd(A)
    p = vh[argmin(s), :]
    p = p.reshape(3, 4)

    r = np.linalg.inv(K) @ p[0:3, 0:3]
    U, D, VT = np.linalg.svd(r)

    R = U @ VT
    T = (np.linalg.inv(K) @ p[:, 3])/D[0]

    if np.linalg.det(R) < 0:
        R = -R
        T = -T

    # T = -R.T @ C
    C = -R.T @ T

    return R, C