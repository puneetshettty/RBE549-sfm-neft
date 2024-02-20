import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

from scipy.spatial.transform import Rotation 

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
    output_image_path = os.path.normpath(f'ransac_results/{source_img}a{target_img}_inliers.png')
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
    
    lines = cv2.computeCorrespondEpilines(np.array(points1), 1, F)
    lines = lines.reshape(-1,3)
    ep1 = drawlines(img1, lines, points1, points2)

    lines = cv2.computeCorrespondEpilines(np.array(points2), 2, F)
    lines = lines.reshape(-1,3)
    ep2 = drawlines(img2, lines, points2, points1)

    output_image_path1 = os.path.normpath(f'results/epilines/{source_img}a{target_img}_1.png')
    output_image_path2 = os.path.normpath(f'results/epilines/{source_img}a{target_img}_2.png')
    cv2.imwrite(output_image_path1,ep1)
    cv2.imwrite(output_image_path2,ep2)

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
    plt.figure(figsize=(10, 6))
    plt.imshow(img1, cmap='gray')
    plt.scatter(proj_linear_norm[0], proj_linear_norm[1], c='r', marker='o',label='Linear Triangulation', s=2)
    plt.scatter(proj_nonlinear_norm[0], proj_nonlinear_norm[1], c='b', marker='x', label='Nonlinear Triangulation', s =3)
    plt.scatter(points1[:, 0], points1[:, 1], c='g', marker='s', label='Detected Features', s=1)
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Reprojection of 3D Points onto Image Plane')
    plt.legend()
    plt.savefig(f'results/reprojection_{img_number}.png')

def plot_triangulation_comparison(X_linear, X_nonlinear, R1, C1, R2, C2, source, target):
    fig = plt.figure("Triangulation Comparison")
    ax = fig.add_subplot()
    
    # Set axis limits
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
    plt.savefig(f'results/triangulation_{source}a{target}.png')
    plt.close()
