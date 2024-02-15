import cv2
import numpy as np
from scipy.spatial.transform import Rotation 
from scipy.optimize import least_squares


def lossfunction(x, world_points, feature_points, K):
    C = x[0:3]
    quats = x[3:]
    R = Rotation.from_quat(quats).as_matrix()
    P = K @ np.hstack((R, -R @ C.reshape(3, 1)))
    reprojection_error = []
    for i in range(len(world_points)):
        # X = np.concatenate((world_points[i], [1]))
        X = world_points[i].reshape(4, 1)
        u, v = feature_points[i][0], feature_points[i][1]
        X_projected = P @ X
        u_projected = X_projected[0] / X_projected[2]
        v_projected = X_projected[1] / X_projected[2]
        error = np.square(u - u_projected) + np.square(v - v_projected)
        reprojection_error.append(error)

    return np.mean(np.array(reprojection_error).squeeze())



def NonLinearPnP(world_points, feature_points, K, R, C):

    quats = Rotation.from_matrix(R).as_quat()
    x0 = np.concatenate((C, quats))

    optimized_x0 = least_squares(lossfunction, x0, method='trf', args=(world_points, feature_points, K))
    optimized_C = optimized_x0.x[0:3]
    optimized_quats = optimized_x0.x[3:]
    R = Rotation.from_quat(optimized_quats).as_matrix()

    return R, optimized_C