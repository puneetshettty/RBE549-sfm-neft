import sys
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

from scipy.spatial.transform import Rotation 

POINTS_DIR = 'results/00_points/'
RANSAC_DIR = 'results/01_ransac/'
EPILINES_DIR = 'results/02_epilines/'
INITIAL_TRIANGULATION_DIR = 'results/03_initial_triangulation/'
TRIANGULATION_COMP_DIR = 'results/04_triangulation_comparison/'
REPROJECTION_DIR = 'results/05_reprojection/'
FINAL_POINT_CLOUD = 'results/06_point_cloud/'

def make_output_dir():
    paths = [
        POINTS_DIR,
        RANSAC_DIR,
        EPILINES_DIR,
        INITIAL_TRIANGULATION_DIR,
        TRIANGULATION_COMP_DIR,
        REPROJECTION_DIR,
        FINAL_POINT_CLOUD,
    ]
    for path in paths:
        if not os.path.exists(path):
            os.makedirs(path)

def plot_points(points, pair, img_number):
    img_path = os.path.normpath(f'./P3Data/{img_number}.png')
    img = cv2.imread(img_path)
    for pt in points:
        color=(0,0,0)
        pt = map(int, pt)
        img = cv2.circle(img,tuple(pt),2,color,-1)

    cv2.imwrite(f'{POINTS_DIR}{img_number}_{pair}.png', img)


def plot_ransac_results(source_img, target_img, inlier_points1, inlier_points2):
    img1_path = os.path.normpath(f'./P3Data/{source_img}.png')
    img2_path = os.path.normpath(f'./P3Data/{target_img}.png')
    img1 = cv2.imread(img1_path)
    img2 = cv2.imread(img2_path)

    # Convert inlier points to KeyPoint objects
    kp1 = [cv2.KeyPoint(pt[0], pt[1], 1) for pt in inlier_points1]
    kp2 = [cv2.KeyPoint(pt[0], pt[1], 1) for pt in inlier_points2]
    
    dmatches = [cv2.DMatch(i, i,0) for i in range(len(inlier_points1))]
    # Draw matches
    output_image_path = os.path.normpath(f'{RANSAC_DIR}{source_img}a{target_img}_inliers.png')
    output_image = cv2.drawMatches(img1, kp1, img2, kp2, dmatches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    
    # Save the resulting image
    cv2.imwrite(output_image_path, output_image)

def plot_epi_lines(source_img, target_img, points1, points2, F):
    def drawlines(img1,lines,pts1,pts2):
        r,c,m = np.shape(img1)

        for r,pt1,pt2 in zip(lines,pts1,pts2):
            color = tuple(np.random.randint(0,255,3).tolist())
            x0,y0 = map(int, [0, -r[2]/r[1] ])
            x1,y1 = map(int, [c, -(r[2]+r[0]*c)/r[1] ])

            pt1 = tuple(map(int, pt1))
            pt2 = tuple(map(int, pt2))

            img1 = cv2.line(img1, (x0,y0), (x1,y1), color,1)
            img1 = cv2.circle(img1,tuple(pt1),2,color,-1)
        return img1
    img1_path = os.path.normpath(f'./P3Data/{source_img}.png')
    img2_path = os.path.normpath(f'./P3Data/{target_img}.png')
    img1 = cv2.imread(img1_path)
    img2 = cv2.imread(img2_path)

    ones = np.ones((1, len(points1)))
    points1_h = np.vstack((np.array(points1).T, ones))
    points2_h = np.vstack((np.array(points2).T, ones))
    
    lines1 = (F @ points1_h).T
    lines2 = (F.T @ points2_h).T

    # lines1 = cv2.computeCorrespondEpilines(np.array(points2), 2, F)
    # lines1 = lines1.reshape(-1,3)
    ep1 = drawlines(img1, lines1, points1, points2)

    # lines2 = cv2.computeCorrespondEpilines(np.array(points1), 1, F)
    # lines2 = lines2.reshape(-1,3)
    ep2 = drawlines(img2, lines2, points2, points1)

    # output_image_path1 = os.path.normpath(f'{EPILINES_DIR}{source_img}a{target_img}_1.png')
    # output_image_path2 = os.path.normpath(f'{EPILINES_DIR}{source_img}a{target_img}_2.png')
    # cv2.imwrite(output_image_path1,ep1)
    # cv2.imwrite(output_image_path2,ep2)

    output_image_path = os.path.normpath(f'{EPILINES_DIR}{source_img}a{target_img}.png')

    new_img = cv2.hconcat([ep1, ep2])

    cv2.imwrite(output_image_path,new_img)


def plot_initial_triangulation(C_stack, R_stack, X_stack, source, target):
    fig = plt.figure("Triangulation Comparison")
    ax = fig.add_subplot()
    
    # Set axis limits
    ax.set_xlim([-30, 30])
    ax.set_ylim([-10, 30])
    
    # Plot linear and nonlinear triangulated points
    colors = ["b", "r", "g", "y"]
    for c,r,x, color in zip(C_stack, R_stack, X_stack, colors):
        ax.scatter(x[:, 0], x[:, 2], c=color, s=1, label="Linear Triangulation")
    

        # angles = Rotation.from_matrix(r).as_euler("XYZ")
        # angles_deg = np.rad2deg(angles)
        # ax.plot(c[0], c[2], marker=(3, 0, int(angles_deg[1])), markersize=15, linestyle='None') 

        
    # Set legend and display plot
    ax.legend()
    plt.savefig(f'{INITIAL_TRIANGULATION_DIR}{source}a{target}.png')
    plt.close()


def plot_triangulation_comparison(X_linear, X_nonlinear, R1, C1, R2, C2, source, target):
    fig = plt.figure("Triangulation Comparison")
    ax = fig.add_subplot()
    
    # # Set axis limits
    ax.set_xlim([-30, 30])
    ax.set_ylim([-10, 30])
    
    # Plot linear and nonlinear triangulated points
    ax.scatter(X_linear[:, 0], X_linear[:, 2], c="b", s=1, label="Linear Triangulation")
    ax.scatter(X_nonlinear[:, 0], X_nonlinear[:, 2], c="r", s=1, label="Nonlinear Triangulation")
    
    # Draw cameras
    for rotation, position, label in [(R1, C1, "1"), (R2, C2, "2")]:
        # Convert rotation matrix to Euler angles
        angles = Rotation.from_matrix(rotation).as_euler("XYZ")
        angles_deg = np.rad2deg(angles)
        
        # Plot camera position
        ax.plot(position[0], position[2], marker=(3, 0, int(angles_deg[1])), markersize=15, linestyle='None') 
        
        # Annotate camera with label
        correction = -0.1
        ax.annotate(label, xy=(position[0] + correction, position[2] + correction))
    
    # Set legend and display plot
    ax.legend()
    plt.savefig(f'{TRIANGULATION_COMP_DIR}{source}a{target}.png')
    plt.close()


def plot_reprojection(X_linear, X_nonlinear, points1, img1, C, R, K, img_number):
    # Compute the camera projection matrix
    points1 = np.array(points1)
    X_linear = np.concatenate((X_linear, np.ones((X_linear.shape[0], 1))), axis=1)
    # X_nonlinear = np.concatenate((X_nonlinear, np.ones((X_nonlinear.shape[0], 1))), axis=1)
    C = C.reshape(3, 1)
    I = np.eye(3)
    P = np.dot(K, np.dot(R, np.hstack((I, -C))))

    # Project 3D points onto the image plane
    proj_linear = np.dot(P, X_linear.T)
    proj_nonlinear = np.dot(P, X_nonlinear.T)

    # Normalize homogeneous coordinates
    proj_linear_norm = proj_linear[:2] / proj_linear[2]
    proj_nonlinear_norm = proj_nonlinear[:2] / proj_nonlinear[2]

    # Plot the image with reprojections
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot()
    ax.set_xlim([0, 800])
    ax.set_ylim([600, 0])
    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
    ax.imshow(img1)
    ax.scatter(proj_linear_norm[0], proj_linear_norm[1], c='r', marker='o',label='Linear Triangulation', s=2)
    ax.scatter(proj_nonlinear_norm[0], proj_nonlinear_norm[1], c='b', marker='x', label='Nonlinear Triangulation', s =3)
    ax.scatter(points1[:, 0], points1[:, 1], c='g', marker='s', label='Detected Features', s=1)
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Reprojection of 3D Points onto Image Plane')
    plt.legend()
    plt.savefig(f'{REPROJECTION_DIR}{img_number}.png')



def plot_point_cloud(X, R_stack, C_stack):
    fig = plt.figure("Final Point Cloud X,Z")
    ax = fig.add_subplot()
    
    # # Set axis limits
    ax.set_xlim([-30, 30])
    ax.set_ylim([-10, 30])
    
    # Plot linear and nonlinear triangulated points
    ax.scatter(X[:, 0], X[:, 2], c="b", s=1, label="Final")
    
    # Draw cameras
    # for rotation, position, label in zip() [(R1, C1, "1"), (R2, C2, "2")]:
    #     # Convert rotation matrix to Euler angles
    #     angles = Rotation.from_matrix(rotation).as_euler("XYZ")
    #     angles_deg = np.rad2deg(angles)
        
    #     # Plot camera position
    #     ax.plot(position[0], position[2], marker=(3, 0, int(angles_deg[1])), markersize=15, linestyle='None') 
        
    #     # Annotate camera with label
    #     correction = -0.1
    #     ax.annotate(label, xy=(position[0] + correction, position[2] + correction))
    
    # Set legend and display plot
    ax.legend()
    plt.savefig(f'{FINAL_POINT_CLOUD}final_point_cloudXZ.png')
    plt.close()

    fig = plt.figure("Final Point Cloud X,Y")
    ax = fig.add_subplot()
    
    # # Set axis limits
    ax.set_xlim([-30, 30])
    ax.set_ylim([-10, 30])
    
    # Plot linear and nonlinear triangulated points
    ax.scatter(X[:, 0], X[:, 1], c="b", s=1, label="Final")
    
    # Draw cameras
    # for rotation, position, label in zip() [(R1, C1, "1"), (R2, C2, "2")]:
    #     # Convert rotation matrix to Euler angles
    #     angles = Rotation.from_matrix(rotation).as_euler("XYZ")
    #     angles_deg = np.rad2deg(angles)
        
    #     # Plot camera position
    #     ax.plot(position[0], position[2], marker=(3, 0, int(angles_deg[1])), markersize=15, linestyle='None') 
        
    #     # Annotate camera with label
    #     correction = -0.1
    #     ax.annotate(label, xy=(position[0] + correction, position[2] + correction))
    
    # Set legend and display plot
    ax.legend()
    plt.savefig(f'{FINAL_POINT_CLOUD}final_point_cloudXY.png')
    plt.close()