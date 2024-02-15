import numpy as np
from scipy.optimize import least_squares

def LossFunction(X, points1, points2, P1, P2):
    x1 = points1[0]
    y1 = points1[1]
    x2 = points2[0]
    y2 = points2[1]
    x1_hat = (P1[0] @ X)/ (P1[2] @ X)
    y1_hat = (P1[1] @ X)/ (P1[2] @ X)
    x2_hat = (P2[0] @ X)/ (P2[2] @ X)
    y2_hat = (P2[1] @ X)/ (P2[2] @ X)

    error = np.square(x1 - x1_hat) + np.square(y1 - y1_hat) + np.square(x2 - x2_hat) + np.square(y2 - y2_hat)
    return error
        

def NonlinearTriangulation(K,C1, R1,C2,R2, points1, points2, Xstack):
    points1 = np.array(points1)
    points2 = np.array(points2)
    # points1 = np.concatenate((points1, np.ones((points1.shape[0], 1))), axis=1)
    # points2 = np.concatenate((points2, np.ones((points2.shape[0], 1))), axis=1)
    Xstack = np.concatenate((Xstack, np.ones((Xstack.shape[0], 1))), axis=1)

    C1 = C1.reshape(3, 1)
    C2 = C2.reshape(3, 1)
    T1 = -R1 @ C1
    T2 = -R2 @ C2
    I = np.eye(3)
    P1 = np.dot(K, np.dot(R1, np.hstack((I, -C1))))  # the P is written this way but not as P = K[R T] because the C and R here means the rotation and translation between cameras 
    P2 = np.dot(K, np.dot(R2, np.hstack((I, -C2))))

    optimal_X = []

    for i, Xpoint in enumerate(Xstack):
        optimal_params = least_squares(fun=LossFunction, x0=Xpoint, method='trf', args=(points1[i], points2[i], P1, P2))
        opt_X = optimal_params.x
        # opt_X = opt_X / opt_X[3]  # Normalize by the homogeneous coordinate
        # opt_X = opt_X[:3]  # Remove the homogeneous coordinate
        optimal_X.append(opt_X)

    # optimal_X = np.vstack(optimal_X)

    return np.array(optimal_X)
    
