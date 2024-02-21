import cv2
import numpy as np
import matplotlib.pyplot as plt
from numpy import ndarray, trace
from numpy.linalg import det, inv
from numpy.linalg import matrix_rank as rank
from scipy.linalg import null_space

def EstimateFundamentalMatrix(pts1,pts2):

    # Converting the match points into (x1,y1) and (x2,y2) 
    A =[]
    for index in range(len(pts1)):

        x1 = pts1[index][0]
        y1 = pts1[index][1]
        x2 = pts2[index][0]
        y2 = pts2[index][1]

        A.append([x1*x2, y1*x2, x2, x1*y2, y1*y2, y2, x1, y1, 1])
        # A.append([x1*x2, x1*y2, x1, y1*x2, y1*y2, y1, x2, y2, 1])


    U,S,VT = np.linalg.svd(A)

    F = VT.T[:,-1]
    
    F = F.reshape(3,3)

    # Emphasising Rank 2
    u, s, vt = np.linalg.svd(F)
    s[2] = 0
    s = np.diag(s)                    
    F = u @ s @ vt

    return F


def convert_to_homogeneous( left_pts, right_pts):
    ones_arr = np.ones((left_pts.shape[0], 1))

    left_pts_h = np.dstack((left_pts, ones_arr))
    right_pts_h = np.dstack((right_pts, ones_arr))

    return left_pts_h, right_pts_h


def get_7pts(left_pts, right_pts):
    random_indices = np.random.randint(0, len(left_pts), 7)
    X = np.matmul(
        right_pts[random_indices].reshape(-1, 3, 1), left_pts[random_indices]
    ).reshape(7, 9)

    return X


def apply_fundamental_matrix( left_pts, right_pts, F,) :
    x_prime_F = np.matmul(right_pts, F)
    norm_coefficient = np.sqrt(
        x_prime_F[:, 0][:, 0] ** 2 + x_prime_F[:, 0][:, 1] ** 2
    ).ravel()
    inliers = abs(x_prime_F @ (left_pts.reshape(-1, 3, 1))).ravel()
    inliers = inliers / norm_coefficient

    return inliers

def get_best_fundamental_matrix( left_pts, right_pts, epsilon=5, n_iter = 1000,):
    best_accuracy = 0
    for _ in range(n_iter):
        # get solution of X system
        X = get_7pts(left_pts, right_pts)
        ker_X = null_space(X)
        F1, F2 = ker_X[:, 0].reshape(3, 3), ker_X[:, 1].reshape(3, 3)

        if rank(F1) == 2:
            inliers = apply_fundamental_matrix(left_pts, right_pts, F1)
            accuracy = np.sum(inliers < epsilon)
            if accuracy > best_accuracy:
                F_best = F1
                best_accuracy = accuracy
                best_inliers = inliers

        elif rank(F2) == 2:
            inliers = apply_fundamental_matrix(left_pts, right_pts, F2)
            accuracy = np.sum(inliers < epsilon)
            if accuracy > best_accuracy:
                F_best = F1
                best_accuracy = accuracy
                best_inliers = inliers

        elif rank(F1) == 3 and rank(F2) == 3:
            # roots for polynomial equation
            p = np.array(
                [
                    det(F1),
                    det(F1) * trace(F2 @ inv(F1)),
                    det(F2) * trace(F1 @ inv(F2)),
                    det(F2),
                ]
            )
            roots = np.roots(p)
            # take only real roots
            real_roots = roots[~np.iscomplex(roots)].real
            for real_root in real_roots:
                F = real_root * F1 + F2
                if rank(F) == 2:
                    inliers = apply_fundamental_matrix(left_pts, right_pts, F)
                    accuracy = np.sum(inliers < epsilon)
                    if accuracy > best_accuracy:
                        F_best = F
                        best_accuracy = accuracy
                        best_inliers = inliers

    return F_best, best_inliers, best_accuracy


def EstimateFundamentalMatrix_7(pts1,pts2):
    f, inliers, acc = get_best_fundamental_matrix(pts1,pts2)
    return f