import cv2
import numpy as np


def LinearTriangulation(points1, points2, K, C1, R1, C2, R2):
    points1 = np.array(points1)
    points2 = np.array(points2)
    points1 = np.concatenate((points1, np.ones((points1.shape[0], 1))), axis=1)
    points2 = np.concatenate((points2, np.ones((points2.shape[0], 1))), axis=1)

    C1 = C1.reshape(3, 1)
    C2 = C2.reshape(3, 1)
    # T1 = - R1 @ C1
    # T2 = - R2 @ C2
    I = np.eye(3)
    P1 = np.dot(K, np.dot(R1, np.hstack((I, -C1))))  # the P is written this way but not as P = K[R T] because the C and R here means the rotation and translation between cameras 
    P2 = np.dot(K, np.dot(R2, np.hstack((I, -C2))))

    X = []

    for pt1, pt2 in zip(points1, points2):
        skew_1 = np.array([[0, -pt1[2], pt1[1]],
                           [pt1[2], 0, -pt1[0]],
                           [-pt1[1], pt1[0], 0]])
        skew_2 = np.array([[0, -pt2[2], pt2[1]],
                            [pt2[2], 0, -pt2[0]],
                            [-pt2[1], pt2[0], 0]])
        
        # A = np.vstack((skew_1 @ P1, skew_2 @ P2))

        u1, v1 = pt1[0:2]
        u2, v2 = pt2[0:2]

        A = np.array([
            v1 * P1[2, :] - P1[1, :],
            P1[0, :] - u1 * P1[2, :],
            v2 * P2[2, :] - P2[1, :],
            P2[0, :] - u2 * P2[2, :],
        ])

        U, S, VT = np.linalg.svd(A)
        
        Xt = VT[-1, :]
        Xt = Xt / Xt[3]
        Xt = Xt[0:3]
                
        X.append(Xt)
    
    # TODO what does this line do?
    X = np.array(X)

    return X
