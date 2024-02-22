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
    p = vh[-1, :]
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


# def LinearPnP(X,points, K):
#     # check if kp1,kp2 and X are homogeneous
#     # points = np.concatenate((points, np.ones((points.shape[0],1))), axis = 1)
#     # X = np.concatenate((X, np.ones((X.shape[0],1))), axis = 1)
    
#     if X.shape[1] == 3:
#         X= np.concatenate((X, np.ones((X.shape[0], 1))), axis=1)
#     else:
#         X = X
    
#     if points.shape[1]==2:
#         points = np.concatenate((points, np.ones((points.shape[0], 1))), axis=1)
#     else:
#         points = points
#     #check if number of points are equal to number of X
#     assert len(points) == len(X)
#     A = []

#     # normalize points
#     # points_normalized = (np.linalg.inv(K) @ points.T).T
#     points_normalized = points
#     for i in range(len(points)):
#         u,v,_ = points_normalized[i]
#         x,y,z,_ = X[i]
#         A.append([x, y, z, 1, 0, 0, 0, 0, -u*x, -u*y, -u*z, -u])
#         A.append([0, 0, 0, 0, x, y, z, 1, -v*x, -v*y, -v*z, -v])
    
#     A = np.array(A)
#     U, D, V = np.linalg.svd(A)
#     P = V[-1, :].reshape(3, 4)
#     temp = np.linalg.inv(K) @ P[:, :3]
#     U_, D_, V_ = np.linalg.svd(temp)
#     R = U_ @ V_
#     C = np.linalg.inv(K) @ P[:, 3]/D_[0]
#     if np.linalg.det(R) < 0:
#         R = -R
#         C = -C
#     C=C.reshape(3,1)
#     C = -R.T@C
#     return R,C


