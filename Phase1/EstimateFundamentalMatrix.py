import cv2
import numpy as np
import matplotlib.pyplot as plt
from numpy import ndarray, trace
from numpy.linalg import det, inv
from numpy.linalg import matrix_rank as rank
from scipy.linalg import null_space
from pprint import pprint
import os

# def normalize(uv):
#     """
#     https://www.cc.gatech.edu/classes/AY2016/cs4476_fall/results/proj3/html/sdai30/index.html
#     """
#     uv_ = np.mean(uv, axis=0)
#     u_,v_ = uv_[0], uv_[1]
#     u_cap, v_cap = uv[:,0] - u_, uv[:,1] - v_

#     s = (2/np.mean(u_cap**2 + v_cap**2))**(0.5)
#     T_scale = np.diag([s,s,1])
#     T_trans = np.array([[1,0,-u_],[0,1,-v_],[0,0,1]])
#     T = T_scale.dot(T_trans)

#     x_ = np.column_stack((uv, np.ones(len(uv))))  #[x,y,1]
#     x_norm = (T.dot(x_.T)).T     #x_ = T.x

#     return  x_norm, T





def EstimateFundamentalMatrix(pts1,pts2):
        


    # Comverting the match points into (x1,y1) and (x2,y2) 

   
    A =[]

    for index in range(len(pts1)):

        x1 = pts1[index][0]
        y1 = pts1[index][1]
        x2 = pts2[index][0]
        y2 = pts2[index][1]

        A.append([x1*x2, y1*x2, x2, x1*y2, y1*y2, y2, x1, y1, 1])


    U,S,VT = np.linalg.svd(A)

    F = VT.T[:,-1]
    
    F = F.reshape(3,3)

    # Emphasising Rank 2
    u, s, vt = np.linalg.svd(F)
    s[2] = 0
    s = np.diag(s)                    
    F = u @ s @ vt


    return F

def get_matches(img_left: ndarray, img_right: ndarray) -> tuple[ndarray, ndarray]:
    """
    Run cv2.SIFT with cv2.BFMatcher.
    Calculate matches.

    Args:
        img_left: Left image. (Source image).
                 Shape (Any, Any).
        img_right: Right image. (Destination image).
                 Shape (Any, Any).

    Returns:
        A tuple containing left and right points.
        Each ndarray has a shape of (Any, 1, 2).
    """
    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(img_left, None)
    kp2, des2 = sift.detectAndCompute(img_right, None)

    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)
    # take only good matches
    good = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            good.append(m)

    left_pts = np.array([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    right_pts = np.array([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

    return left_pts, right_pts


def convert_to_homogeneous(
    left_pts: ndarray, right_pts: ndarray
) -> tuple[ndarray, ndarray]:
    """
    Add ones to make all coordinates homogeneous

    Args:
        left_pts: Points on the left image.
                 Shape: (Any, 1, 2).
        right_pts: Points on the right image.
                 Shape: (Any, 1, 2).

    Returns:
        A tuple containing homogeneous points on the left and right images.
        Each ndarray has a shape of (Any, 1, 3).
    """
    ones_arr = np.ones((left_pts.shape[0], 1))

    left_pts_h = np.dstack((left_pts, ones_arr))
    right_pts_h = np.dstack((right_pts, ones_arr))

    return left_pts_h, right_pts_h


def get_7pts(left_pts: ndarray, right_pts: ndarray) -> ndarray:
    """
    Get random 7 points from left_pts, right_pts points.

    Args:
        left_pts: Points on the left image. (homogeneous).
                 Shape: (Any, 1, 3).
        right_pts: Points on the right image. (homogeneous).
                 Shape: (Any, 1, 3).

    Returns:
        Random 7 points. Shape: (7, 9).
    """
    random_indices = np.random.randint(0, len(left_pts), 7)
    X = np.matmul(
        right_pts[random_indices].reshape(-1, 3, 1), left_pts[random_indices]
    ).reshape(7, 9)

    return X


def apply_fundamental_matrix(
    left_pts: ndarray,
    right_pts: ndarray,
    F: ndarray,
) -> ndarray:
    """
    Get inliers points by applying fundamental matrix to the points.
    Formula  sum_(|x`Fx|/norm_coefficient)

    Args:
        left_pts: Points on the left image. (homogeneous).
                 Shape: (Any, 1, 3).
        right_pts: Points on the right image. (homogeneous).
                 Shape: (Any, 1, 3).
        F: Fundamental matrix.
           Shape: (3, 3).

    Returns:
        Inliers array.
        Shape: (Any, ).
    """
    x_prime_F = np.matmul(right_pts, F)
    norm_coefficient = np.sqrt(
        x_prime_F[:, 0][:, 0] ** 2 + x_prime_F[:, 0][:, 1] ** 2
    ).ravel()
    inliers = abs(x_prime_F @ (left_pts.reshape(-1, 3, 1))).ravel()
    inliers = inliers / norm_coefficient

    return inliers


def get_best_fundamental_matrix(
    left_pts: ndarray,
    right_pts: ndarray,
    epsilon: int,
    n_iter: int,
) -> tuple[ndarray, ndarray, int]:
    """
    Calculate the best fundamental matrix.
    Use number of inliers as an accuracy criterion.

    Args:
        left_pts: Points on the left image. (homogeneous).
                 Shape: (Any, 1, 3).
        right_pts: Points on the right image. (homogeneous).
                 Shape: (Any, 1, 3).
        epsilon: Threshold parameter in pixels.
        n_iter: Number of iterations.

    Returns:
        A tuple:
            The best fundamental matrix. Shape: (3, 3).
            Inliers array. Shape: (Any, ).
            Number of inliers (the best accuracy).
    """
    best_accuracy = 0
    for _ in range(n_iter):
        # get solution of X system
        X = get_7pts(left_pts, right_pts)
        ker_X = null_space(X)
        F1, F2 = ker_X[:, 0].reshape(3, 3), ker_X[:, 1].reshape(3, 3)

        if rank(F1) == 2:
            # print("F1 rank 2")
            inliers = apply_fundamental_matrix(left_pts, right_pts, F1)
            accuracy = np.sum(inliers < epsilon)
            if accuracy > best_accuracy:
                F_best = F1
                best_accuracy = accuracy
                best_inliers = inliers

        elif rank(F2) == 2:
            # print("F2 rank 2")
            inliers = apply_fundamental_matrix(left_pts, right_pts, F2)
            accuracy = np.sum(inliers < epsilon)
            if accuracy > best_accuracy:
                F_best = F1
                best_accuracy = accuracy
                best_inliers = inliers

        elif rank(F1) == 3 and rank(F2) == 3:
            # print("F1 and F2 rank 3")
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


def calculate_epipolar_lines(points: ndarray, F: ndarray, whichImage: str) -> ndarray:
    """
    Calculate epipolar lines by applying fundamental matrix to points.

    Args:
        points: Points on the left or right image.
                 Shape: (Any, 1, 2).
        F: Fundamental matrix.
           Shape: (3, 3).
        whichImage: "left" or "right".

    Returns:
        Lines array.
        Shape: (Any, 3).
    """
    assert whichImage in [
        "left",
        "right",
    ], 'whichImage argument must be either "left" or "right"'

    ones_arr = np.ones((points.shape[0], 1))
    points = np.dstack((points, ones_arr))

    if whichImage == "left":
        lines = np.matmul(F, points.reshape(-1, 3, 1)).reshape(-1, 3)

    if whichImage == "right":
        lines = np.matmul(F.T, points.reshape(-1, 3, 1)).reshape(-1, 3)

    return lines


def draw_lines_and_points(img, lines, points):
    """
    Draw lines and points on the image.

    Args:
        img: Shape:  (Any, Any).
        lines: Corresponding epipolar lines.
        points: Points for the image.

    Returns:
        RGB image with lines. Shape: (Any, Any, 3).
    """

    r, c = img.shape
    img_with_lines = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    for r, pt in zip(lines, points):
        color = tuple(np.random.randint(0, 255, 3).tolist())

        x0, y0 = map(int, [0, -r[2] / r[1]])
        x1, y1 = map(int, [c, -(r[2] + r[0] * c) / r[1]])

        img_with_lines = cv2.line(img_with_lines, (x0, y0), (x1, y1), color, 1)
        img_with_lines = cv2.circle(
            img_with_lines, tuple(pt.ravel().astype(np.int64)), 5, color, -1
        )

    return img_with_lines


def draw_epipolar_lines(
    left_pts: ndarray,
    right_pts: ndarray,
    img_left: ndarray,
    img_right: ndarray,
    F: ndarray,
    inliers: ndarray,
    epsilon: int,
) -> tuple[ndarray, ndarray]:
    """
    Draw epipolar lines from the right image
    and inliers from the left image
    on the left image.

    Draw epipolar lines from the left image
    and inliers from the right image
    on the right image.

    Args:
        left_pts: Points on the left image.
                 Shape: (Any, 1, 2).
        right_pts: Points on the right image.
                 Shape: (Any, 1, 2).
        img_left: Left image.
                 Shape: (Any, Any).
        img_right: Right image.
                 Shape: (Any, Any).
        inliers: Inliers array.
                 Shape: (Any, ).
        epsilon: Threshold parameter in pixels.

    Returns:
        A tuple containing left and right images with epipolar lines.
        Each ndarray has a shape of (Any, Any, 3).
    """
    mask = inliers < epsilon

    left_inliers = left_pts[mask == 1]
    right_inliers = right_pts[mask == 1]

    right_lines_on_the_left = calculate_epipolar_lines(
        points=right_inliers, F=F, whichImage="right"
    )
    left_img_with_epiliens = draw_lines_and_points(
        img=img_left, lines=right_lines_on_the_left, points=left_inliers
    )

    left_lines_on_the_right = calculate_epipolar_lines(
        points=left_inliers, F=F, whichImage="left"
    )
    right_img_with_epiliens = draw_lines_and_points(
        img=img_right, lines=left_lines_on_the_right, points=right_inliers
    )

    return left_img_with_epiliens, right_img_with_epiliens , left_inliers, right_inliers



def EstimateFundamentalMatrix_(source_img, target_img, pts1,pts2):
    img1_path = os.path.normpath(f'./P3Data/{source_img}.png')
    img2_path = os.path.normpath(f'./P3Data/{target_img}.png')
    img_left = cv2.imread(img1_path, cv2.IMREAD_GRAYSCALE)
    img_right = cv2.imread(img2_path, cv2.IMREAD_GRAYSCALE)

    assert img_left.shape[:2] == img_right.shape[:2], "Images dimensions do not match."

    left_pts, right_pts = get_matches(img_left=img_left, img_right=img_right)

    src_pts_h, dst_pts_h = convert_to_homogeneous(left_pts, right_pts)
    # left_pts = np.array(pts1).reshape(-1,1,3)
    # right_pts = np.array(pts2).reshape(-1,1,3)
    F_best, best_inliers, best_accuracy = get_best_fundamental_matrix(
        left_pts=src_pts_h,
        right_pts=dst_pts_h,
        epsilon=5,
        n_iter=10,
    )

    mask = best_inliers < 5
    left_inliers = left_pts[mask == 1]
    right_inliers = right_pts[mask == 1]

    # left_img_with_epiliens, right_img_with_epiliens, left_inliers, right_inliers = draw_epipolar_lines(
    #     left_pts=left_pts.copy(),
    #     right_pts=right_pts.copy(),
    #     img_left=img_left,
    #     img_right=img_right,
    #     F=F_best,
    #     inliers=best_inliers,
    #     epsilon=5,
    # )

    # # save results
    # cv2.imwrite("left_img_with_right_epipolar_liens.png", left_img_with_epiliens)
    # cv2.imwrite("right_img_with_left_epipolar_liens.png", right_img_with_epiliens)

    # F = cv2.findFundamentalMat(pts1, pts2, cv2.FM_RANSAC, 0.1, 0.99)[0]
    # print('F from cv2: ')
    # pprint(F[0])
    # print('F from our implementation: ')
    # pprint(f)
    return F_best, left_inliers, right_inliers




    # """
    # tested BAD
    # input:
    #     v1 - N x 3
    #     v2 - N x 3
    # output:
    #     F - fundamental matrix 3 x 3
    # """
    # v1 = np.array(v1)
    # v2 = np.array(v2)
    # v1 = v1[:,:2] # N x 3
    # v2 = v2[:,:2] # N x 3
    # # construct Ax = 0
    # x1, y1 = v1[:,0], v1[:,1] # N,
    # x2, y2 = v2[:,0], v2[:,1] # N,
    # ones = np.ones(x1.shape[0])

    # A = [x1*x2, y1*x2, x2, x1*y2, y1*y2, y2, x1, y1, ones] # N x 9
    # A = np.vstack(A).T # N x 9

    # # get SVD of A
    # U,sigma,V = np.linalg.svd(A) # N x N, N x 9, 9 x 9
    # f = V[np.argmin(sigma),:] # 9,

    # # reconstruct F from singular vector
    # F = f.reshape((3,3))
    # #F = F/f[8]

    # # take SVD of F
    # UF, sigmaF, VF = np.linalg.svd(F)
    # sigmaF[2] = 0 # enforcing rank 2 constraint
    # reestimatedF = UF @ np.diag(sigmaF) @ VF

    # return reestimatedF




    # Testing Bad Mandeep
    # normalised = True
    # pts1 = np.array(pts1)
    # pts2 = np.array(pts2)
    # pts1 = pts1[:,:2] # N x 3
    # pts2 = pts2[:,:2] # N x 3

    # x1,x2 = pts1, pts2

    # if x1.shape[0] > 7:
    #     if normalised == True:
    #         x1_norm, T1 = normalize(x1)
    #         x2_norm, T2 = normalize(x2)
    #     else:
    #         x1_norm,x2_norm = x1,x2
            
    #     A = np.zeros((len(x1_norm),9))     #Fundamental matrix 3x3 so 9columns and min 8 points So rows>=8 columns = 9
    #     for i in range(0, len(x1_norm)):
    #         x_1,y_1 = x1_norm[i][0], x1_norm[i][1]
    #         x_2,y_2 = x2_norm[i][0], x2_norm[i][1]
    #         A[i] = np.array([x_1*x_2, x_2*y_1, x_2, y_2*x_1, y_2*y_1, y_2, x_1, y_1, 1])

    #     U, S, VT = np.linalg.svd(A, full_matrices=True)
    #     F = VT.T[:, -1]
    #     F = F.reshape(3,3)

    #     u, s, vt = np.linalg.svd(F)
    #     s = np.diag(s)
    #     s[2,2] = 0                     #Due to Noise F can be full rank i.e 3, but we need to make it rank 2 by assigning zero to last diagonal element and thus we get the epipoles
    #     F = np.dot(u, np.dot(s, vt))
        
    #     if normalised:
    #         F = np.dot(T2.T, np.dot(F, T1))   #This is given in algorithm for normalization
    #         F = F / F[2,2]
    #     return F

    # else:
    #     return None
