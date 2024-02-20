import os
import sys
import glob
import random

import cv2
import numpy as np
import matplotlib.pyplot as plt
import itertools
from pprint import pprint
from GetInliersRANSAC import inlier_ransac
from EssentialMatrixFromFundamentalMatrix import EssentialMatrixFromFundamentalMatrix
from ExtractCameraPose import ExtractCameraPose
from LinearTriangulation import LinearTriangulation
from DisambiguateCameraPose import DisambiguateCameraPose
from NonlinearTriangulation import NonlinearTriangulation
from PnPRANSAC import PnPRANSAC
from NonLinearPnP import NonLinearPnP
from PlotResults import plot_ransac_results,plot_reprojection,plot_triangulation_comparison, plot_epi_lines


def projectionMatrix(R,C,K):
    C = np.reshape(C,(3,1))
    I = np.identity(3)
    P = np.dot(K,np.dot(R,np.hstack((I,-C))))
    return P

def ReProjectionError(X,pts1, pts2, C1, R1, C2, R2, K):
    X = np.array(X)
    if X.shape[1] == 3:
        X = np.concatenate((X, np.ones((X.shape[0], 1))), axis=1)
    p1 = projectionMatrix(R1, C1, K)
    p2 = projectionMatrix(R2, C2, K)
    error = []

    for pt1, pt2, X_3d in zip(pts1,pts2,X):

        # print("This i s p1",p1,p1.shape)
        # print("This i s p2",p2,p2.shape)
        p1_1T, p1_2T, p1_3T = p1
        # print(p1_1T.shape, p1_2T.shape, p1_3T.shape )
        p1_1T, p1_2T, p1_3T = p1_1T.reshape(1,4), p1_2T.reshape(1,4), p1_3T.reshape(1,4)

        p2_1T, p2_2T, p2_3T = p2
        p2_1T, p2_2T, p2_3T = p2_1T.reshape(1,4), p2_2T.reshape(1,4), p2_3T.reshape(1,4)

        X_3d = X_3d.reshape(4,1)

        "Reprojection error w.r.t 1st Ref camera points"
        u1, v1 = pt1[0], pt1[1]
        # print(u1,v1)
        # print(p1_1T.shape,(p1_3T.shape),X.shape)
        u1_projection = np.divide(p1_1T.dot(X_3d), p1_3T.dot(X_3d))
        v1_projection = np.divide(p1_2T.dot(X_3d), p1_3T.dot(X_3d))
        err1 = np.square(v1 - v1_projection) + np.square(u1 - u1_projection)

        "Reprojection error w.r.t 2nd Ref camera points"
        u2, v2 = pt2[0], pt2[1]
        u2_projection = np.divide(p2_1T.dot(X_3d), p2_3T.dot(X_3d))
        v2_projection = np.divide(p2_2T.dot(X_3d), p2_3T.dot(X_3d))
        err2 = np.square(v2 - v2_projection) + np.square(u2 - u2_projection)

        err = err1 + err2
        error.append(err)

    return np.mean(error)

def reprojectionErrorPnP(x3D, pts, K, R, C):
    P = projectionMatrix(R,C,K)
    # print("P :",P)
    
    Error = []
    for X, pt in zip(x3D, pts):

        p_rows = np.vsplit(P, 3) # Split P into rows
        p_1T, p_2T, p_3T = p_rows # rows of P
        p_1T, p_2T, p_3T = p_1T.reshape(1,-1), p_2T.reshape(1,-1), p_3T.reshape(1,-1)
        X = X.reshape(-1,1)
        # X = np.hstack((X, np.ones((X.shape[0], 1)))).reshape(-1,1) # make X it a column of homogenous vector
        ## reprojection error for reference camera points 
        u, v = pt[0], pt[1]
        u_proj = np.divide(p_1T.dot(X) , p_3T.dot(X))
        v_proj =  np.divide(p_2T.dot(X) , p_3T.dot(X))

        E = np.square(v - v_proj) + np.square(u - u_proj)

        Error.append(E)

    mean_error = np.mean(np.array(Error).squeeze())
    return mean_error


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
                l = 0
                for line in file:
                    l += 1
                    if i == 0:
                        num_features = int(line.split(':')[1].strip())
                        i += 1
                        continue
                    data = line.split(' ')
                    feature_coords = (float(data[4]), float(data[5]))
                    num_matches = int(data[0])
                    num_pairs = num_matches - 1

                    img_ids = [source_img]
                    matching_coords = [feature_coords]


                    for j in range(num_pairs):
                        img_id = int(data[6 + 3 * j])
                        matching_coord = (float(data[6 + 3 * j + 1]), float(data[6 + 3 * j + 2]))

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

                    # if num_pairs > 1:
                    #     combinations_3 = list(itertools.combinations(combinations, 3))
                        
                    #     # Only doing for 3 images
                    #     for x,y,z in combinations_3:
                    #         key = f"{x[0]}a{y[0]}a{z[0]}"

                    #         if key in matches:
                    #             matches[key].append((x[1], y[1], z[1]))
                    #         else: 
                    #             matches[key] = [(x[1], y[1], z[1])]
                    # if l == 6:
                    #     break
    return matches 




class Main:
    def __init__(self, files_path):
        # Example usage:
        self.files_path = files_path

        matches = read_matching_file(self.files_path)

        # Convert dictionary values to sets with inner elements converted to tuples
        matches_as_sets = {key: {tuple(inner_tuple) for inner_tuple in value} for key, value in matches.items()}

        # Convert dictionary values back to lists with inner elements converted to lists
        self.matches = {key: [list(inner_tuple) for inner_tuple in value] for key, value in matches_as_sets.items()}

        # self.target_pairs = ['1a2','1a3','1a4','1a5']
        self.target_pairs = ['1a2']

    def estimate_fundamental_matrices(self):
        self.fundamental_matrices = {}

        indexes_dict = {}  # Dictionary to store indexes for each pair
        inlier_indexes_dict = {}

        self.pairwise_inlier_points_1 = {}
        self.pairwise_inlier_points_2 = {}

        # for pair in image_pairs:
        for pair in self.target_pairs:
            source_img, target_img = map(int, [pair[0], pair[2]])
            points1 = []
            points2 = []
            indexes_key = f"{source_img}a{target_img}"  # Key for the indexes list

            if indexes_key not in indexes_dict:
                indexes_dict[indexes_key] = []  # Initialize the list if not present

            for i in range(len(self.matches[pair])):
                points1.append(self.matches[pair][i][0])
                points2.append(self.matches[pair][i][1])
                indexes_dict[indexes_key].append(i)  # Append the index to the corresponding list
            

            F, inlier_points1, inlier_points2, inlier_indexes_dict[indexes_key] = inlier_ransac(points1, points2, indexes_dict[indexes_key], 1000, 0.1)
            self.fundamental_matrices[indexes_key] = F
            # Load images
            self.pairwise_inlier_points_1[pair] = inlier_points1
            self.pairwise_inlier_points_2[pair] = inlier_points2
            plot_ransac_results(source_img, target_img, inlier_points1, inlier_points2)

            plot_epi_lines(source_img, target_img, points1, points2, F)


    def drawlines(self,img1,lines,pts1,pts2):
        ''' img1 - image on which we draw the epilines for the points in img2
        lines - corresponding epilines '''
        r,c,m = np.shape(img1)
        print(r,c)

        for r,pt1,pt2 in zip(lines,pts1,pts2):
            color = tuple(np.random.randint(0,255,3).tolist())
            x0,y0 = map(int, [0, -r[2]/r[1] ])
            x1,y1 = map(int, [c, -(r[2]+r[0]*c)/r[1] ])

            pt1 = tuple(map(int, pt1))
            pt2 = tuple(map(int, pt2))

            img1 = cv2.line(img1, (x0,y0), (x1,y1), color,1)
            img1 = cv2.circle(img1,tuple(pt1),2,color,-1)
        return img1

    def estimate_essential_matrix(self):
        # Getting Camera Calibration K from calibration.txt
        K = np.loadtxt(os.path.join(self.files_path, 'calibration.txt'))
        K = np.array(K.reshape(3,3))
        self.K = K

        #essential matrix from fundamental matrix and k
        self.essential_matrix = {}
        for image_pair in self.target_pairs:
            self.essential_matrix[image_pair] = EssentialMatrixFromFundamentalMatrix(K, self.fundamental_matrices[image_pair])

    def extract_camera_pose(self):
        self.ambigous_camera_poses = {}
        for image_pair in self.target_pairs:
            self.ambigous_camera_poses[image_pair] = ExtractCameraPose(self.essential_matrix[image_pair])

    def triangulate_and_fix_camera_pose(self):
        C1 = np.array([0, 0, 0])
        R1 = np.eye(3)

        for pair in self.target_pairs:
            source_img, target_img = map(int, [pair[0], pair[2]])
            X_stack = []
            C_stack, R_stack = self.ambigous_camera_poses[pair]
            for i in range(len(C_stack)):
                C2 = C_stack[i]
                R2 = R_stack[i]
                Xt = LinearTriangulation(
                    self.pairwise_inlier_points_1[pair],
                    self.pairwise_inlier_points_2[pair],
                    self.K, C1, R1, C2, R2)
                X_stack.append(Xt)

            C_linear, R_linear, X_linear = DisambiguateCameraPose(C_stack, R_stack, X_stack)

            # Nonlinear Triangulation
            X_nonlinear = NonlinearTriangulation(self.K, C1, R1, C_linear, R_linear, 
                                                 self.pairwise_inlier_points_1[pair],
                                                 self.pairwise_inlier_points_2[pair],
                                                 X_linear)
    
            plot_triangulation_comparison(X_linear,X_nonlinear,R_linear, C_linear,R1,C1, source_img, target_img)


            img1_path = f'./P3Data/{source_img}.png'
            img2_path = f'./P3Data/{target_img}.png'
            img1 = cv2.imread(img1_path)
            img2 = cv2.imread(img2_path)
            img_number = source_img
            plot_reprojection(X_linear, X_nonlinear, self.pairwise_inlier_points_1[pair], img1, C_linear, R_linear, self.K, img_number)

files_path = os.path.normpath('P3Data')
main = Main(files_path)
main.estimate_fundamental_matrices()
main.estimate_essential_matrix()
main.extract_camera_pose()
main.triangulate_and_fix_camera_pose()


# X_positive = []
# index_positive = []

# for X,index in zip(X_nonlinear, linear_indexes):
    
#     if (np.isnan(X[0])) | (X[0] <= 0):
#         # print(f"Point {index} is behind the camera")
#         continue
#     else:
#         # print(f"Point {index} is in front of the camera")
#         X_positive.append(X)
#         index_positive.append(index)



# # print(f"Number of points in front of the camera: {len(X_positive)}")
# # print(f"Number of points behind the camera: {len(X_linear) - len(X_positive)}")
        
# # extracting features of the given index points
# points1_positive = []
# # points2_positive = []

# for index in index_positive:
#     points1_positive.append(points1[index])
#     # points2_positive.append(points2[index])



# #linear PnP


# R_lpnp, C_lpnp = PnPRANSAC(X_positive, points1_positive, K, 10000, 200)
# # print(f'Rotation matrix from PnP: {R_pnp}')
# # print(f'Camera position from PnP: {C_pnp}')
# # print(f'camera position from linear triangulation: {C_linear}')


# # #___________________________________________________________________Error Calculation_______________________________________________________

# error_lpnp = reprojectionErrorPnP(X_positive, points1_positive, K, R_lpnp, C_lpnp)
# print(f"Mean reprojection error for PnP: {np.mean(error_lpnp)}")
# # error_linear = ReProjectionError(X_linear, inlier_points1, inlier_points2, C1, R1, C_linear, R_linear, K)
# # print(f"Mean reprojection error for linear triangulation: {np.mean(error_linear)}")
# # error_nonlinear = ReProjectionError(X_nonlinear, inlier_points1, inlier_points2, C1, R1, C_linear, R_linear, K)
# # print(f"Mean reprojection error for nonlinear triangulation: {np.mean(error_nonlinear)}")

# # #___________________________________________________________________________________________________________________________________________

# R_nlpnp, C_nlpnp = NonLinearPnP(X_positive, points1_positive, K, R_lpnp, C_lpnp)

# error_nlpnp = reprojectionErrorPnP(X_positive, points1_positive, K, R_nlpnp, C_nlpnp)
# print(f"Mean reprojection error for Nonlinear PnP: {np.mean(error_nlpnp)}")
