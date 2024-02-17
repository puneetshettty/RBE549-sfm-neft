import os
import glob
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
    num_png_files = len(png_files)
    
    matches = {}  # Create an empty dictionary for matches

    for file in txt_files:
        if 'matching' in file:
            matches_path = file
            source_img = matches_path.split("matching")[1].split(".")[0]  # Get the source image number from the file name
            source_img = int(source_img)

            for i in range(1, min(6, num_png_files - source_img + 1)):
                key = f"{source_img}a{source_img + i}"  # Dynamically generate the key
                matches[key] = set()

            # Now you have a dictionary with keys like '1a2', '1a3', etc. for each source image
            # You can use these keys to store the matches for each source image
            # print(matches)  # Print for demonstration, you can replace this with your actual processing logic
                

            with open(matches_path, 'r') as file:
                i = 0
                for line in file:
                    if i == 0:
                        num_features = int(line.split(':')[1].strip())
                        i += 1
                    else:
                        data = line.split(' ')
                        feature_coords = [float(data[4]), float(data[5])]
                        num_matches = int(data[0])
                        num_pairs = num_matches - 1

                        for j in range(num_pairs):
                            img_id = int(data[6 + 3 * j])
                            matches_coords = [float(data[6 + 3 * j + 1]), float(data[6 + 3 * j + 2])]
                            feature_matches = (tuple(feature_coords), tuple(matches_coords))  # Create a tuple for each pair of matches
                            key = f"{source_img}a{img_id}"
                            matches[key].add(feature_matches)  # Append the matches to the list for the corresponding key

                            # # If there are more than 1 pair involved, create keys for combinations of all image pairs
                            if num_pairs > 1:
                                for comb in itertools.combinations(range(source_img + 1, min(6, source_img + num_png_files)), num_pairs):
                                    comb_key = f"{source_img}a{'a'.join(str(c) for c in comb)}"
                                    matches[comb_key] = set()

            
                    i += 1
                    if i == 12:
                        break

    return matches 


# Example usage:
files_path = '.\P3Data'
matches = read_matching_file(files_path)
image_pairs = list(matches.keys())
image_pairs_1 = [key for key in image_pairs if len(key) <= len('1a2')]
print(image_pairs_1)
# convert matches[key] to list
for key in matches:
    matches[key] = list(matches[key])

for key in image_pairs_1:
    # print(f'key1: {key}')
    for i in range(len(matches[key])):
        m = matches[key][i][0]
        targets = [matches[key][i][1]]
        targets_id = []
        source_img = int(key.split('a')[0])
        for k, key1 in enumerate(image_pairs_1):
            # print(k)
            if k+1 < (len(image_pairs_1) - 1):            
                for key2 in image_pairs_1[k+1:]:
                    print(f'key1: {key1}')
                    print(f'key2: {key2}')
                    for j in range(len(matches[key2])):
                        m1 = matches[key2][j][0]
                        if m == m1:
                            targets.append(matches[key2][j][1])
                            targets_id.append(int(key2.split('a')[1]))


        if len(targets) > 1:
            comb_key = f"{source_img}a{'a'.join(str(id) for id in targets_id)}"
            print(comb_key)
            if comb_key in matches.keys():
                matches[comb_key].append(targets)
                # print(f'Made key {comb_key} with {len(targets)} targets')








image_pairs = list(matches.keys())
# print(len(matches['1a2a3']))
