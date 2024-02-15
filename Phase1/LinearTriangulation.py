import cv2
import numpy as np


def LinearTriangulation(points1, points2, K, C1, R1, C2, R2):
    points1 = np.array(points1)
    points2 = np.array(points2)
    points1 = np.concatenate((points1, np.ones((points1.shape[0], 1))), axis=1)
    points2 = np.concatenate((points2, np.ones((points2.shape[0], 1))), axis=1)

    C1 = C1.reshape(3, 1)
    C2 = C2.reshape(3, 1)
    T1 = - R1 @ C1
    T2 = - R2 @ C2
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
        
        A = np.vstack((skew_1 @ P1, skew_2 @ P2))

        U, S, VT = np.linalg.svd(A)
        
        Xt = VT[np.argmin(S), :]
        Xt = Xt / Xt[3]
        Xt = Xt[0:3]
        
        X.append(Xt)
    
    X = np.vstack(X)

    return X


# def main():
#     K = np.array([[568.996140852, 0, 643.21055941],
#                   [0, 568.988362396, 477.982801038],
#                   [0, 0, 1]])
#     C1 = np.array([0, 0, 0])
#     R1 = np.eye(3)
#     C2 = np.array([0.34920214, 0.46360747, 0.81432547])
#     R2 = np.array([[-0.85043032,  0.21032695,  0.48221451],
#                     [ 0.25479772, -0.63725701,  0.72731123],
#                     [ 0.46026773,  0.74139469,  0.48835186]])
#     pts1 = [[36.3025, 170.408], [37.0963, 167.092], [42.2047, 200.851], [42.8895, 170.28], [43.4867, 191.131], [47.8875, 196.202], [49.336, 203.336], [49.6084, 542.71], [56.786, 202.413]]
#     pts2 = [[10.2986, 150.262], [10.9623, 146.818], [19.3666, 183.084], [17.6202, 150.441], [19.588, 172.699], [24.261, 178.192], [26.2842, 185.986], [220.36, 515.917], [34.2992, 184.947]]
#     X = LinearTriangulation(pts1, pts2, K, C1, R1, C2, R2)
#     print(X)


# if __name__ == "__main__":
#     main()