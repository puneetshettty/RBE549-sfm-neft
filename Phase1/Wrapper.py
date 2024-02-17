import sys
import os
import glob
from pprint import pprint

import cv2
import numpy as np
import matplotlib.pyplot as plt
import itertools
from GetInliersRANSAC import inlier_ransac
from EssentialMatrixFromFundamentalMatrix import EssentialMatrixFromFundamentalMatrix
from ExtractCameraPose import ExtractCameraPose
from LinearTriangulation import LinearTriangulation
from DisambiguateCameraPose import DisambiguateCameraPose
from NonlinearTriangulation import NonlinearTriangulation
from PnPRANSAC import PnPRANSAC
from NonLinearPnP import NonLinearPnP
from scipy.spatial.transform import Rotation 


# def projectionMatrix(R,C,K):
#     C = np.reshape(C,(3,1))
#     I = np.identity(3)
#     P = np.dot(K,np.dot(R,np.hstack((I,-C))))
#     return P

# def ReProjectionError(X,pts1, pts2, C1, R1, C2, R2, K):
#     X = np.array(X)
#     if X.shape[1] == 3:
#         X = np.concatenate((X, np.ones((X.shape[0], 1))), axis=1)
#     p1 = projectionMatrix(R1, C1, K)
#     p2 = projectionMatrix(R2, C2, K)
#     error = []

#     for pt1, pt2, X_3d in zip(pts1,pts2,X):

#         # print("This i s p1",p1,p1.shape)
#         # print("This i s p2",p2,p2.shape)
#         p1_1T, p1_2T, p1_3T = p1
#         # print(p1_1T.shape, p1_2T.shape, p1_3T.shape )
#         p1_1T, p1_2T, p1_3T = p1_1T.reshape(1,4), p1_2T.reshape(1,4), p1_3T.reshape(1,4)

#         p2_1T, p2_2T, p2_3T = p2
#         p2_1T, p2_2T, p2_3T = p2_1T.reshape(1,4), p2_2T.reshape(1,4), p2_3T.reshape(1,4)

#         X_3d = X_3d.reshape(4,1)

#         "Reprojection error w.r.t 1st Ref camera points"
#         u1, v1 = pt1[0], pt1[1]
#         # print(u1,v1)
#         # print(p1_1T.shape,(p1_3T.shape),X.shape)
#         u1_projection = np.divide(p1_1T.dot(X_3d), p1_3T.dot(X_3d))
#         v1_projection = np.divide(p1_2T.dot(X_3d), p1_3T.dot(X_3d))
#         err1 = np.square(v1 - v1_projection) + np.square(u1 - u1_projection)

#         "Reprojection error w.r.t 2nd Ref camera points"
#         u2, v2 = pt2[0], pt2[1]
#         u2_projection = np.divide(p2_1T.dot(X_3d), p2_3T.dot(X_3d))
#         v2_projection = np.divide(p2_2T.dot(X_3d), p2_3T.dot(X_3d))
#         err2 = np.square(v2 - v2_projection) + np.square(u2 - u2_projection)

#         err = err1 + err2
#         error.append(err)

#     return np.mean(error)

# def reprojectionErrorPnP(x3D, pts, K, R, C):
#     P = projectionMatrix(R,C,K)
#     # print("P :",P)
    
#     Error = []
#     for X, pt in zip(x3D, pts):

#         p_rows = np.vsplit(P, 3) # Split P into rows
#         p_1T, p_2T, p_3T = p_rows # rows of P
#         p_1T, p_2T, p_3T = p_1T.reshape(1,-1), p_2T.reshape(1,-1), p_3T.reshape(1,-1)
#         X = X.reshape(-1,1)
#         # X = np.hstack((X, np.ones((X.shape[0], 1)))).reshape(-1,1) # make X it a column of homogenous vector
#         ## reprojection error for reference camera points 
#         u, v = pt[0], pt[1]
#         u_proj = np.divide(p_1T.dot(X) , p_3T.dot(X))
#         v_proj =  np.divide(p_2T.dot(X) , p_3T.dot(X))

#         E = np.square(v - v_proj) + np.square(u - u_proj)

#         Error.append(E)

#     mean_error = np.mean(np.array(Error).squeeze())
#     return mean_error


# def plot_reprojection(X_linear, X_nonlinear, points1, img1, C, R, K, img_number):
#     # Compute the camera projection matrix
#     points1 = np.array(points1)
#     X_linear = np.concatenate((X_linear, np.ones((X_linear.shape[0], 1))), axis=1)
#     # X_nonlinear = np.concatenate((X_nonlinear, np.ones((X_nonlinear.shape[0], 1))), axis=1)
#     C = C.reshape(3, 1)
#     I = np.eye(3)
#     P = np.dot(K, np.dot(R, np.hstack((I, -C))))

#     # Project 3D points onto the image plane
#     proj_linear = np.dot(P, X_linear.T)
#     proj_nonlinear = np.dot(P, X_nonlinear.T)

#     # Normalize homogeneous coordinates
#     proj_linear_norm = proj_linear[:2] / proj_linear[2]
#     proj_nonlinear_norm = proj_nonlinear[:2] / proj_nonlinear[2]

#     # Plot the image with reprojections
#     plt.figure(figsize=(10, 6))
#     plt.imshow(img1, cmap='gray')
#     plt.scatter(proj_linear_norm[0], proj_linear_norm[1], c='r', marker='o',label='Linear Triangulation', s=2)
#     plt.scatter(proj_nonlinear_norm[0], proj_nonlinear_norm[1], c='b', marker='x', label='Nonlinear Triangulation', s =3)
#     plt.scatter(points1[:, 0], points1[:, 1], c='g', marker='s', label='Detected Features', s=1)
#     plt.xlabel('X')
#     plt.ylabel('Y')
#     plt.title('Reprojection of 3D Points onto Image Plane')
#     plt.legend()
#     plt.savefig(f'.\\results\\reprojection_{img_number}.png')

# def plot_triangulation_comparison(X_linear, X_nonlinear, R1, C1, R2, C2, source, target):
#     fig = plt.figure("Triangulation Comparison")
#     ax = fig.add_subplot()
    
#     # Set axis limits
#     ax.set_xlim([-30, 30])
#     ax.set_ylim([-10, 30])
    
#     # Plot linear and nonlinear triangulated points
#     ax.scatter(X_linear[:, 0], X_linear[:, 2], c="b", s=1, label="Linear Triangulation")
#     ax.scatter(X_nonlinear[:, 0], X_nonlinear[:, 2], c="r", s=1, label="Nonlinear Triangulation")
    
#     # Draw cameras
#     for rotation, position, label in [(R1, C1, "1"), (R2, C2, "2")]:
#         # Convert rotation matrix to Euler angles
#         angles = Rotation.from_matrix(rotation).as_euler("XYZ")
#         angles_deg = np.rad2deg(angles)
        
#         # Plot camera position
#         ax.plot(position[0], position[2], marker=(3, 0, int(angles_deg[1])), markersize=15, linestyle='None') 
        
#         # Annotate camera with label
#         correction = -0.1
#         ax.annotate(label, xy=(position[0] + correction, position[2] + correction))
    
#     # Set legend and display plot
#     ax.legend()
#     plt.savefig(f'.\\results\\triangulation_{source}a{target}.png')
#     plt.close()



def read_matching_file(files_path):
    txt_files = glob.glob(os.path.join(files_path, '*.txt'))
    png_files = glob.glob(os.path.join(files_path, '*.png'))
    
    matches = {}  # Create an empty dictionary for matches

    for file in txt_files:
        if 'matching' in file:
            matches_path = file
            source_img = matches_path.split("matching")[1].split(".")[0]  # Get the source image number from the file name
            source_img = int(source_img)

            with open(matches_path, 'r') as file:
                i = 0
                for line in file:
                    if i == 0:
                        num_features = int(line.split(':')[1].strip())
                        i += 1
                        continue
                    data = line.split(' ')
                    feature_coords = [float(data[4]), float(data[5])]
                    num_matches = int(data[0])
                    num_pairs = num_matches - 1

                    img_ids = [source_img]
                    matching_coords = [feature_coords]


                    for j in range(num_pairs):
                        img_id = int(data[6 + 3 * j])
                        matching_coord = [float(data[6 + 3 * j + 1]), float(data[6 + 3 * j + 2])]

                        img_ids.append(img_id)
                        matching_coords.append(matching_coord)

                    # sort the matches according to image id
                    # Makes sure that for a given combination, only one key is created
                    img_ids, matching_coords = zip(*sorted(zip(img_ids, matching_coords))) 
                    combinations = list(zip(img_ids, matching_coords))

                    combinations_2 = list(itertools.combinations(combinations, 2))
                    for x, y in combinations_2:
                        key = f"{x[0]}a{y[0]}"
                        if key in matches:
                            matches[key].append((x[1], y[1]))
                        else: 
                            matches[key] = [(x[1], y[1])]

                    if num_pairs > 1:
                        combinations_3 = list(itertools.combinations(combinations, 3))
                        
                        # Only doing for 3 images
                        for x,y,z in combinations_3:
                            key = f"{x[0]}a{y[0]}a{z[0]}"

                            if key in matches:
                                matches[key].append((x[1], y[1], z[1]))
                            else: 
                                matches[key] = [(x[1], y[1], z[1])]


    return matches 


# Example usage:
files_path = os.path.normpath('P3Data')

matches = read_matching_file(files_path)
pprint(matches)

pprint([(key, len(matches[key])) for key in matches])